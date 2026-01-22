                        stay_to_uncompress -= 14;
                    }

                    /* copy data to uncompress buffer (B44 block can exceed target resolution)*/
                    index_tl_x = iX * 4;
                    index_tl_y = iY * 4;

                    for (y = index_tl_y; y < FFMIN(index_tl_y + 4, td->ysize); y++) {
                        for (x = index_tl_x; x < FFMIN(index_tl_x + 4, td->xsize); x++) {
                            index_out = target_channel_offset * td->xsize + y * td->channel_line_size + 2 * x;
                            index_tmp = (y-index_tl_y) * 4 + (x-index_tl_x);
                            td->uncompressed_data[index_out] = tmp_buffer[index_tmp] & 0xff;
                            td->uncompressed_data[index_out + 1] = tmp_buffer[index_tmp] >> 8;
                        }
                    }
                }
            }
            target_channel_offset += 2;
        } else {/* Float or UINT 32 channel */
            if (stay_to_uncompress < td->ysize * td->xsize * 4) {
                av_log(s, AV_LOG_ERROR, "Not enough data for uncompress channel: %d", stay_to_uncompress);
                return AVERROR_INVALIDDATA;
            }

            for (y = 0; y < td->ysize; y++) {
                index_out = target_channel_offset * td->xsize + y * td->channel_line_size;
                memcpy(&td->uncompressed_data[index_out], sr, td->xsize * 4);
                sr += td->xsize * 4;
            }
            target_channel_offset += 4;

            stay_to_uncompress -= td->ysize * td->xsize * 4;
        }
    }

    return 0;
}

static int decode_block(AVCodecContext *avctx, void *tdata,
                        int jobnr, int threadnr)
{
    EXRContext *s = avctx->priv_data;
    AVFrame *const p = s->picture;
    EXRThreadData *td = &s->thread_data[threadnr];
    const uint8_t *channel_buffer[4] = { 0 };
    const uint8_t *buf = s->buf;
    uint64_t line_offset, uncompressed_size;
    uint8_t *ptr;
    uint32_t data_size;
    int line, col = 0;
    uint64_t tile_x, tile_y, tile_level_x, tile_level_y;
    const uint8_t *src;
    int step = s->desc->flags & AV_PIX_FMT_FLAG_FLOAT ? 4 : 2 * s->desc->nb_components;
    int bxmin = 0, axmax = 0, window_xoffset = 0;
    int window_xmin, window_xmax, window_ymin, window_ymax;
    int data_xoffset, data_yoffset, data_window_offset, xsize, ysize;
    int i, x, buf_size = s->buf_size;
    int c, rgb_channel_count;
    float one_gamma = 1.0f / s->gamma;
    avpriv_trc_function trc_func = avpriv_get_trc_function_from_trc(s->apply_trc_type);
    int ret;

    line_offset = AV_RL64(s->gb.buffer + jobnr * 8);

    if (s->is_tile) {
        if (buf_size < 20 || line_offset > buf_size - 20)
            return AVERROR_INVALIDDATA;

        src  = buf + line_offset + 20;
        if (s->is_multipart)
            src += 4;

        tile_x = AV_RL32(src - 20);
        tile_y = AV_RL32(src - 16);
        tile_level_x = AV_RL32(src - 12);
        tile_level_y = AV_RL32(src - 8);

        data_size = AV_RL32(src - 4);
        if (data_size <= 0 || data_size > buf_size - line_offset - 20)
            return AVERROR_INVALIDDATA;

        if (tile_level_x || tile_level_y) { /* tile level, is not the full res level */
            avpriv_report_missing_feature(s->avctx, "Subres tile before full res tile");
            return AVERROR_PATCHWELCOME;
        }

        line = s->ymin + s->tile_attr.ySize * tile_y;
        col = s->tile_attr.xSize * tile_x;

        if (line < s->ymin || line > s->ymax ||
            s->xmin + col  < s->xmin ||  s->xmin + col  > s->xmax)
            return AVERROR_INVALIDDATA;

        td->ysize = FFMIN(s->tile_attr.ySize, s->ydelta - tile_y * s->tile_attr.ySize);
        td->xsize = FFMIN(s->tile_attr.xSize, s->xdelta - tile_x * s->tile_attr.xSize);

        if (td->xsize * (uint64_t)s->current_channel_offset > INT_MAX)
            return AVERROR_INVALIDDATA;

        td->channel_line_size = td->xsize * s->current_channel_offset;/* uncompress size of one line */
        uncompressed_size = td->channel_line_size * (uint64_t)td->ysize;/* uncompress size of the block */
    } else {
        if (buf_size < 8 || line_offset > buf_size - 8)
            return AVERROR_INVALIDDATA;

        src  = buf + line_offset + 8;
        if (s->is_multipart)
            src += 4;
        line = AV_RL32(src - 8);

        if (line < s->ymin || line > s->ymax)
            return AVERROR_INVALIDDATA;

        data_size = AV_RL32(src - 4);
        if (data_size <= 0 || data_size > buf_size - line_offset - 8)
            return AVERROR_INVALIDDATA;

        td->ysize          = FFMIN(s->scan_lines_per_block, s->ymax - line + 1); /* s->ydelta - line ?? */
        td->xsize          = s->xdelta;

        if (td->xsize * (uint64_t)s->current_channel_offset > INT_MAX)
            return AVERROR_INVALIDDATA;

        td->channel_line_size = td->xsize * s->current_channel_offset;/* uncompress size of one line */
        uncompressed_size = td->channel_line_size * (uint64_t)td->ysize;/* uncompress size of the block */

        if ((s->compression == EXR_RAW && (data_size != uncompressed_size ||
                                           line_offset > buf_size - uncompressed_size)) ||
            (s->compression != EXR_RAW && (data_size > uncompressed_size ||
                                           line_offset > buf_size - data_size))) {
            return AVERROR_INVALIDDATA;
        }
    }

    window_xmin = FFMIN(avctx->width, FFMAX(0, s->xmin + col));
    window_xmax = FFMIN(avctx->width, FFMAX(0, s->xmin + col + td->xsize));
    window_ymin = FFMIN(avctx->height, FFMAX(0, line ));
    window_ymax = FFMIN(avctx->height, FFMAX(0, line + td->ysize));
    xsize = window_xmax - window_xmin;
    ysize = window_ymax - window_ymin;

    /* tile or scanline not visible skip decoding */
    if (xsize <= 0 || ysize <= 0)
        return 0;

    /* is the first tile or is a scanline */
    if(col == 0) {
        window_xmin = 0;
        /* pixels to add at the left of the display window */
        window_xoffset = FFMAX(0, s->xmin);
        /* bytes to add at the left of the display window */
        bxmin = window_xoffset * step;
    }

    /* is the last tile or is a scanline */
    if(col + td->xsize == s->xdelta) {
        window_xmax = avctx->width;
         /* bytes to add at the right of the display window */
        axmax = FFMAX(0, (avctx->width - (s->xmax + 1))) * step;
    }

    if (data_size < uncompressed_size || s->is_tile) { /* td->tmp is use for tile reorganization */
        av_fast_padded_malloc(&td->tmp, &td->tmp_size, uncompressed_size);
        if (!td->tmp)
            return AVERROR(ENOMEM);
    }

    if (data_size < uncompressed_size) {
        av_fast_padded_malloc(&td->uncompressed_data,
                              &td->uncompressed_size, uncompressed_size + 64);/* Force 64 padding for AVX2 reorder_pixels dst */

        if (!td->uncompressed_data)
            return AVERROR(ENOMEM);

        ret = AVERROR_INVALIDDATA;
        switch (s->compression) {
        case EXR_ZIP1:
        case EXR_ZIP16:
            ret = zip_uncompress(s, src, data_size, uncompressed_size, td);
            break;
        case EXR_PIZ:
            ret = piz_uncompress(s, src, data_size, uncompressed_size, td);
            break;
        case EXR_PXR24:
            ret = pxr24_uncompress(s, src, data_size, uncompressed_size, td);
            break;
        case EXR_RLE:
            ret = rle_uncompress(s, src, data_size, uncompressed_size, td);
            break;
        case EXR_B44:
        case EXR_B44A:
            ret = b44_uncompress(s, src, data_size, uncompressed_size, td);
            break;
        }
        if (ret < 0) {
            av_log(avctx, AV_LOG_ERROR, "decode_block() failed.\n");
            return ret;
        }
        src = td->uncompressed_data;
    }

    /* offsets to crop data outside display window */
    data_xoffset = FFABS(FFMIN(0, s->xmin + col)) * (s->pixel_type == EXR_HALF ? 2 : 4);
    data_yoffset = FFABS(FFMIN(0, line));
    data_window_offset = (data_yoffset * td->channel_line_size) + data_xoffset;

    if (!s->is_luma) {
        channel_buffer[0] = src + (td->xsize * s->channel_offsets[0]) + data_window_offset;
        channel_buffer[1] = src + (td->xsize * s->channel_offsets[1]) + data_window_offset;
        channel_buffer[2] = src + (td->xsize * s->channel_offsets[2]) + data_window_offset;
        rgb_channel_count = 3;
    } else { /* put y data in the first channel_buffer */
        channel_buffer[0] = src + (td->xsize * s->channel_offsets[1]) + data_window_offset;
        rgb_channel_count = 1;
    }
     if (s->channel_offsets[3] >= 0)
        channel_buffer[3] = src + (td->xsize * s->channel_offsets[3]) + data_window_offset;

    if (s->desc->flags & AV_PIX_FMT_FLAG_FLOAT) {

        /* todo: change this when a floating point pixel format with luma with alpha is implemented */
        int channel_count = s->channel_offsets[3] >= 0 ? 4 : rgb_channel_count;
        if (s->is_luma) {
            channel_buffer[1] = channel_buffer[0];
            channel_buffer[2] = channel_buffer[0];
        }

        for (c = 0; c < channel_count; c++) {
            int plane = s->desc->comp[c].plane;
            ptr = p->data[plane] + window_ymin * p->linesize[plane] + (window_xmin * 4);

            for (i = 0; i < ysize; i++, ptr += p->linesize[plane]) {
                const uint8_t *src;
                union av_intfloat32 *ptr_x;

                src = channel_buffer[c];
                ptr_x = (union av_intfloat32 *)ptr;

                // Zero out the start if xmin is not 0
                memset(ptr_x, 0, bxmin);
                ptr_x += window_xoffset;

                if (s->pixel_type == EXR_FLOAT) {
                    // 32-bit
                    union av_intfloat32 t;
                    if (trc_func && c < 3) {
                        for (x = 0; x < xsize; x++) {
                            t.i = bytestream_get_le32(&src);
                            t.f = trc_func(t.f);
                            *ptr_x++ = t;
                        }
                    } else if (one_gamma != 1.f) {
                        for (x = 0; x < xsize; x++) {
                            t.i = bytestream_get_le32(&src);
                            if (t.f > 0.0f && c < 3)  /* avoid negative values */
                                t.f = powf(t.f, one_gamma);
                            *ptr_x++ = t;
                        }
                    } else {
                        for (x = 0; x < xsize; x++) {
                            t.i = bytestream_get_le32(&src);
                            *ptr_x++ = t;
                        }
                    }
                } else if (s->pixel_type == EXR_HALF) {
                    // 16-bit
                    if (c < 3 || !trc_func) {
                        for (x = 0; x < xsize; x++) {
                            *ptr_x++ = s->gamma_table[bytestream_get_le16(&src)];
                        }
                    } else {
                        for (x = 0; x < xsize; x++) {
                            *ptr_x++ = exr_half2float(bytestream_get_le16(&src));;
                        }
                    }
                }

                // Zero out the end if xmax+1 is not w
                memset(ptr_x, 0, axmax);
                channel_buffer[c] += td->channel_line_size;
            }
        }
    } else {

        av_assert1(s->pixel_type == EXR_UINT);
        ptr = p->data[0] + window_ymin * p->linesize[0] + (window_xmin * s->desc->nb_components * 2);

        for (i = 0; i < ysize; i++, ptr += p->linesize[0]) {

            const uint8_t * a;
