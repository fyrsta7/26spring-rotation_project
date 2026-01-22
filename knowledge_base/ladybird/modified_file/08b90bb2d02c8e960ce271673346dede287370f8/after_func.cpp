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

    auto x_scale = reference_frame.x_scale;
    auto y_scale = reference_frame.x_scale;
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
    auto reference_frame_width = reference_frame.size.width() >> subsampling_x;
    auto reference_frame_buffer_at = [&](u32 row, u32 column) -> u16 const& {
        return reference_frame_buffer[row * reference_frame_width + column];
    };

    auto block_buffer_at = [&](u32 row, u32 column) -> u16& {
        return block_buffer[row * width + column];
    };

    // The variable lastX is set equal to ( (RefFrameWidth[ refIdx ] + subX) >> subX) - 1.
    // The variable lastY is set equal to ( (RefFrameHeight[ refIdx ] + subY) >> subY) - 1.
    // (lastX and lastY specify the coordinates of the bottom right sample of the reference plane.)
    i32 scaled_right = ((reference_frame.size.width() + subsampling_x) >> subsampling_x) - 1;
    i32 scaled_bottom = ((reference_frame.size.height() + subsampling_y) >> subsampling_y) - 1;

    // The variable intermediateHeight specifying the height required for the intermediate array is set equal to (((h -
    // 1) * yStep + 15) >> 4) + 8.
    static constexpr auto maximum_intermediate_height = (((maximum_block_dimensions - 1) * maximum_scaled_step + 15) >> 4) + 8;
    auto intermediate_height = (((height - 1) * scaled_step_y + 15) >> 4) + 8;
    VERIFY(intermediate_height <= maximum_intermediate_height);
    // The sub-sample interpolation is effected via two one-dimensional convolutions. First a horizontal filter is used
    // to build up a temporary array, and then this array is vertically filtered to obtain the final prediction. The
    // fractional parts of the motion vectors determine the filtering process. If the fractional part is zero, then the
    // filtering is equivalent to a straight sample copy.
    // The filtering is applied as follows:
    // The array intermediate is specified as follows:
    // Note: Height is specified by `intermediate_height`, width is specified by `width`
    Array<u16, maximum_intermediate_height * maximum_block_dimensions> intermediate_buffer;
    auto intermediate_buffer_at = [&](u32 row, u32 column) -> u16& {
        return intermediate_buffer[row * width + column];
    };

    // Check our reference frame bounds before starting the loop.
    reference_frame_buffer_at(scaled_bottom, scaled_right);

    for (auto row = 0u; row < intermediate_height; row++) {
        auto clamped_row = static_cast<size_t>(clip_3(0, scaled_bottom, (offset_scaled_block_y >> 4) + static_cast<i32>(row) - 3));
        u16 const* scan_line = &reference_frame_buffer_at(clamped_row, 0);

        for (auto column = 0u; column < width; column++) {
            auto samples_start = offset_scaled_block_x + static_cast<i32>(scaled_step_x * column);

            i32 accumulated_samples = 0;
            for (auto t = 0u; t < 8u; t++) {
                auto sample = scan_line[clip_3(0, scaled_right, (samples_start >> 4) + static_cast<i32>(t) - 3)];
                accumulated_samples += subpel_filters[block_context.interpolation_filter][samples_start & 15][t] * sample;
            }
            intermediate_buffer_at(row, column) = clip_1(block_context.frame_context.color_config.bit_depth, rounded_right_shift(accumulated_samples, 7));
        }
    }

    for (auto row = 0u; row < height; row++) {
        for (auto column = 0u; column < width; column++) {
            auto samples_start = (offset_scaled_block_y & 15) + static_cast<i32>(scaled_step_y * row);
            auto const* scan_column = &intermediate_buffer_at(samples_start >> 4, column);
            auto const* subpel_filters_for_samples = subpel_filters[block_context.interpolation_filter][samples_start & 15];

            i32 accumulated_samples = 0;
            for (auto t = 0u; t < 8u; t++) {
                auto sample = *scan_column;
                accumulated_samples += subpel_filters_for_samples[t] * sample;
                scan_column += width;
            }
            block_buffer_at(row, column) = clip_1(block_context.frame_context.color_config.bit_depth, rounded_right_shift(accumulated_samples, 7));
