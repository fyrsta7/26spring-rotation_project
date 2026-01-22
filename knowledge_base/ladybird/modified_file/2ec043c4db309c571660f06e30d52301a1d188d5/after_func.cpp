    reference_frame.scaled_step_y = scaled_step_y;

    return {};
}

DecoderErrorOr<void> Decoder::predict_inter_block(u8 plane, BlockContext const& block_context, ReferenceIndex reference_index, u32 block_row, u32 block_column, u32 x, u32 y, u32 width, u32 height, u32 block_index, Span<u16> block_buffer)
{
    VERIFY(width <= maximum_block_dimensions && height <= maximum_block_dimensions);
    // 2. The motion vector selection process in section 8.5.2.1 is invoked with plane, refList, blockIdx as inputs
    // and the output being the motion vector mv.
    auto motion_vector = select_motion_vector(plane, block_context, reference_index, block_index);

    // 3. The motion vector clamping process in section 8.5.2.2 is invoked with plane, mv as inputs and the output
    // being the clamped motion vector clampedMv
    auto clamped_vector = clamp_motion_vector(plane, block_context, block_row, block_column, motion_vector);

    // 4. The motion vector scaling process in section 8.5.2.3 is invoked with plane, refList, x, y, clampedMv as
    // inputs and the output being the initial location startX, startY, and the step sizes stepX, stepY.

    // 8.5.2.3 Motion vector scaling process
    // The inputs to this process are:
    // − a variable plane specifying which plane is being predicted,
    // − a variable refList specifying that we should scale to match reference frame ref_frame[ refList ],
    // − variables x and y specifying the location of the top left sample in the CurrFrame[ plane ] array of the region
    // to be predicted,
    // − a variable clampedMv specifying the clamped motion vector.
    // The outputs of this process are the variables startX and startY giving the reference block location in units of
    // 1/16 th of a sample, and variables xStep and yStep giving the step size in units of 1/16 th of a sample.
    // This process is responsible for computing the sampling locations in the reference frame based on the motion
    // vector. The sampling locations are also adjusted to compensate for any difference in the size of the reference
    // frame compared to the current frame.

    // NOTE: Some of this is done in advance by Decoder::prepare_referenced_frame().

    // A variable refIdx specifying which reference frame is being used is set equal to
    // ref_frame_idx[ ref_frame[ refList ] - LAST_FRAME ].
    auto reference_frame_index = block_context.frame_context.reference_frame_indices[block_context.reference_frame_types[reference_index] - ReferenceFrameType::LastFrame];
    auto const& reference_frame = m_parser->m_reference_frames[reference_frame_index];

    // Scale values range from 8192 to 262144.
    // 16384 = 1:1, higher values indicate the reference frame is larger than the current frame.
    auto x_scale = reference_frame.x_scale;
    auto y_scale = reference_frame.y_scale;

    // The amount of subpixels between each sample of this block. Non-16 values will cause the output to be scaled.
    auto scaled_step_x = reference_frame.scaled_step_x;
    auto scaled_step_y = reference_frame.scaled_step_y;

    // The variable baseX is set equal to (x * xScale) >> REF_SCALE_SHIFT.
    // The variable baseY is set equal to (y * yScale) >> REF_SCALE_SHIFT.
    // (baseX and baseY specify the location of the block in the reference frame if a zero motion vector is used).
    i32 base_x = (x * x_scale) >> REF_SCALE_SHIFT;
    i32 base_y = (y * y_scale) >> REF_SCALE_SHIFT;

    // The variable lumaX is set equal to (plane > 0) ? x << subsampling_x : x.
    // The variable lumaY is set equal to (plane > 0) ? y << subsampling_y : y.
    // (lumaX and lumaY specify the location of the block to be predicted in the current frame in units of luma
    // samples.)
    bool subsampling_x = plane > 0 ? block_context.frame_context.color_config.subsampling_x : false;
    bool subsampling_y = plane > 0 ? block_context.frame_context.color_config.subsampling_y : false;
    i32 luma_x = x << subsampling_x;
    i32 luma_y = y << subsampling_y;

    // The variable fracX is set equal to ( (16 * lumaX * xScale) >> REF_SCALE_SHIFT) & SUBPEL_MASK.
    // The variable fracY is set equal to ( (16 * lumaY * yScale) >> REF_SCALE_SHIFT) & SUBPEL_MASK.
    i32 frac_x = ((16 * luma_x * x_scale) >> REF_SCALE_SHIFT) & SUBPEL_MASK;
    i32 frac_y = ((16 * luma_y * y_scale) >> REF_SCALE_SHIFT) & SUBPEL_MASK;

    // The variable dX is set equal to ( (clampedMv[ 1 ] * xScale) >> REF_SCALE_SHIFT) + fracX.
    // The variable dY is set equal to ( (clampedMv[ 0 ] * yScale) >> REF_SCALE_SHIFT) + fracY.
    // (dX and dY specify a scaled motion vector.)
    i32 scaled_vector_x = ((clamped_vector.column() * x_scale) >> REF_SCALE_SHIFT) + frac_x;
    i32 scaled_vector_y = ((clamped_vector.row() * y_scale) >> REF_SCALE_SHIFT) + frac_y;

    // The output variable startX is set equal to (baseX << SUBPEL_BITS) + dX.
    // The output variable startY is set equal to (baseY << SUBPEL_BITS) + dY.
    i32 offset_scaled_block_x = (base_x << SUBPEL_BITS) + scaled_vector_x;
    i32 offset_scaled_block_y = (base_y << SUBPEL_BITS) + scaled_vector_y;

    // A variable ref specifying the reference frame contents is set equal to FrameStore[ refIdx ].
    auto& reference_frame_buffer = reference_frame.frame_planes[plane];
    auto reference_frame_width = y_size_to_uv_size(subsampling_x, reference_frame.size.width()) + MV_BORDER * 2;

    // The variable lastX is set equal to ( (RefFrameWidth[ refIdx ] + subX) >> subX) - 1.
    // The variable lastY is set equal to ( (RefFrameHeight[ refIdx ] + subY) >> subY) - 1.
    // (lastX and lastY specify the coordinates of the bottom right sample of the reference plane.)
    // Ad-hoc: These variables are not needed, since the reference frame is expanded to contain the samples that
    // may be referenced by motion vectors on the edge of the frame.

    // The sub-sample interpolation is effected via two one-dimensional convolutions. First a horizontal filter is used
    // to build up a temporary array, and then this array is vertically filtered to obtain the final prediction. The
    // fractional parts of the motion vectors determine the filtering process. If the fractional part is zero, then the
    // filtering is equivalent to a straight sample copy.
    // The filtering is applied as follows:

    constexpr auto sample_offset = 3;

    auto subpixel_row_from_reference_row = [offset_scaled_block_y](u32 row) {
        return (offset_scaled_block_y >> SUBPEL_BITS) + static_cast<i32>(row);
    };
    auto reference_index_for_row = [reference_frame_width](i32 row) {
        return static_cast<size_t>(MV_BORDER + row) * reference_frame_width;
    };

    // The variable intermediateHeight specifying the height required for the intermediate array is set equal to (((h -
    // 1) * yStep + 15) >> 4) + 8.
    static constexpr auto maximum_intermediate_height = (((maximum_block_dimensions - 1) * maximum_scaled_step + 15) >> 4) + 8;
    auto const intermediate_height = (((height - 1) * scaled_step_y + 15) >> 4) + 8;
    VERIFY(intermediate_height <= maximum_intermediate_height);
    // Check our reference frame bounds before starting the loop.
    auto const last_possible_reference_index = reference_index_for_row(subpixel_row_from_reference_row(intermediate_height - sample_offset));
    VERIFY(reference_frame_buffer.size() >= last_possible_reference_index);

    VERIFY(block_buffer.size() >= static_cast<size_t>(width) * height);

    auto const reference_block_x = MV_BORDER + (offset_scaled_block_x >> SUBPEL_BITS);
    auto const reference_block_y = MV_BORDER + (offset_scaled_block_y >> SUBPEL_BITS);
    auto const reference_subpixel_x = offset_scaled_block_x & SUBPEL_MASK;
    auto const reference_subpixel_y = offset_scaled_block_y & SUBPEL_MASK;

    // OPTIMIZATION: If the fractional part of a component of the motion vector is 0, we want to do a fast path
    //               skipping one or both of the convolutions.
    bool const copy_x = reference_subpixel_x == 0;
    bool const copy_y = reference_subpixel_y == 0;
    bool const unscaled_x = scaled_step_x == 16;
    bool const unscaled_y = scaled_step_y == 16;

    // The array intermediate is specified as follows:
    // Note: Height is specified by `intermediate_height`, width is specified by `width`
    Array<u16, maximum_intermediate_height * maximum_block_dimensions> intermediate_buffer;
    auto const bit_depth = block_context.frame_context.color_config.bit_depth;
    auto const* reference_start = reference_frame_buffer.data() + reference_block_y * reference_frame_width + reference_block_x;

    // FIXME: We are using 16-bit products to vectorize the filter loops, but when filtering in a high bit-depth video, they will truncate.
    //        Instead of hardcoding them, we should have the bit depth as a template parameter, and the accumulators can select a size based
    //        on whether the bit depth > 8.
    //        Note that we only get a benefit from this on the default CPU target. If we enable AVX2 here, we may want to specialize the
    //        function for the CPU target and remove the cast to i16 so that it doesn't have to truncate on AVX2, where it can do the full
    //        unrolled 32-bit product loops in one vector.

    if (unscaled_x && unscaled_y && bit_depth == 8) {
        if (copy_x && copy_y) {
            // We can memcpy here to avoid doing any real work.
            auto const* reference_scan_line = &reference_frame_buffer[reference_block_y * reference_frame_width + reference_block_x];
            auto* destination_scan_line = block_buffer.data();

            for (auto row = 0u; row < height; row++) {
                memcpy(destination_scan_line, reference_scan_line, width * sizeof(*destination_scan_line));
                reference_scan_line += reference_frame_width;
                destination_scan_line += width;
            }

            return {};
        }

        auto horizontal_convolution_unscaled = [](auto bit_depth, auto* destination, auto width, auto height, auto const* source, auto source_stride, auto filter, auto subpixel_x) {
            source -= sample_offset;
            auto const source_end_skip = source_stride - width;

            for (auto row = 0u; row < height; row++) {
                for (auto column = 0u; column < width; column++) {
                    i32 accumulated_samples = 0;
                    for (auto t = 0; t < 8; t++) {
                        auto sample = source[t];
                        accumulated_samples += static_cast<i16>(subpel_filters[filter][subpixel_x][t] * sample);
                    }

                    *destination = clip_1(bit_depth, rounded_right_shift(accumulated_samples, 7));
                    source++;
                    destination++;
                }
                source += source_end_skip;
            }
        };

        if (copy_y) {
            horizontal_convolution_unscaled(bit_depth, block_buffer.data(), width, height, reference_start, reference_frame_width, block_context.interpolation_filter, reference_subpixel_x);
            return {};
        }

        auto vertical_convolution_unscaled = [](auto bit_depth, auto* destination, auto width, auto height, auto const* source, auto source_stride, auto filter, auto subpixel_y) {
            auto const source_end_skip = source_stride - width;

            for (auto row = 0u; row < height; row++) {
                for (auto column = 0u; column < width; column++) {
                    auto const* scan_column = source;
                    i32 accumulated_samples = 0;
                    for (auto t = 0; t < 8; t++) {
                        auto sample = *scan_column;
                        accumulated_samples += static_cast<i16>(subpel_filters[filter][subpixel_y][t] * sample);
                        scan_column += source_stride;
                    }
                    *destination = clip_1(bit_depth, rounded_right_shift(accumulated_samples, 7));
                    source++;
                    destination++;
                }
                source += source_end_skip;
            }
        };

        if (copy_x) {
            vertical_convolution_unscaled(bit_depth, block_buffer.data(), width, height, reference_start - (sample_offset * reference_frame_width), reference_frame_width, block_context.interpolation_filter, reference_subpixel_y);
            return {};
        }

        horizontal_convolution_unscaled(bit_depth, intermediate_buffer.data(), width, intermediate_height, reference_start - (sample_offset * reference_frame_width), reference_frame_width, block_context.interpolation_filter, reference_subpixel_x);
        vertical_convolution_unscaled(bit_depth, block_buffer.data(), width, height, intermediate_buffer.data(), width, block_context.interpolation_filter, reference_subpixel_y);
        return {};
    }

    // NOTE: Accumulators below are 32-bit to allow high bit-depth videos to decode without overflows.
    //       These should be changed when the accumulators above are.

    auto horizontal_convolution_scaled = [](auto bit_depth, auto* destination, auto width, auto height, auto const* source, auto source_stride, auto filter, auto subpixel_x, auto scale_x) {
        source -= sample_offset;

        for (auto row = 0u; row < height; row++) {
            auto scan_subpixel = subpixel_x;
            for (auto column = 0u; column < width; column++) {
                auto const* scan_line = source + (scan_subpixel >> 4);
                i32 accumulated_samples = 0;
                for (auto t = 0; t < 8; t++) {
                    auto sample = scan_line[t];
                    accumulated_samples += subpel_filters[filter][scan_subpixel & SUBPEL_MASK][t] * sample;
                }

                *destination = clip_1(bit_depth, rounded_right_shift(accumulated_samples, 7));
                destination++;
                scan_subpixel += scale_x;
            }
            source += source_stride;
        }
    };

    auto vertical_convolution_scaled = [](auto bit_depth, auto* destination, auto width, auto height, auto const* source, auto source_stride, auto filter, auto subpixel_y, auto scale_y) {
        for (auto row = 0u; row < height; row++) {
            auto const* source_column_base = source + (subpixel_y >> SUBPEL_BITS) * source_stride;

            for (auto column = 0u; column < width; column++) {
                auto const* scan_column = source_column_base + column;
                i32 accumulated_samples = 0;
                for (auto t = 0; t < 8; t++) {
                    auto sample = *scan_column;
                    accumulated_samples += subpel_filters[filter][subpixel_y & SUBPEL_MASK][t] * sample;
                    scan_column += source_stride;
                }

                *destination = clip_1(bit_depth, rounded_right_shift(accumulated_samples, 7));
                destination++;
            }
            subpixel_y += scale_y;
        }
    };

