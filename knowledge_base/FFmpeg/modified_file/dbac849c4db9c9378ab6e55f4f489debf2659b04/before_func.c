    } else if (esc_count < 8) {
        esc_count -= 4;
        src ++;
        src_size --;
        if (esc_count > 0) {
            /* Zero run coding only, no range coding. */
            for (i = 0; i < height; i++) {
                int res = lag_decode_zero_run_line(l, dst + (i * stride), src,
                                                   src_end, width, esc_count);
                if (res < 0)
                    return res;
                src += res;
            }
        } else {
            if (src_size < width * height)
                return AVERROR_INVALIDDATA; // buffer not big enough
            /* Plane is stored uncompressed */
            for (i = 0; i < height; i++) {
                memcpy(dst + (i * stride), src, width);
                src += width;
            }
        }
    } else if (esc_count == 0xff) {
        /* Plane is a solid run of given value */
        for (i = 0; i < height; i++)
            memset(dst + i * stride, src[1], width);
        /* Do not apply prediction.
           Note: memset to 0 above, setting first value to src[1]
           and applying prediction gives the same result. */
        return 0;
    } else {
        av_log(l->avctx, AV_LOG_ERROR,
               "Invalid zero run escape code! (%#x)\n", esc_count);
        return -1;
    }

    if (l->avctx->pix_fmt != AV_PIX_FMT_YUV422P) {
        for (i = 0; i < height; i++) {
            lag_pred_line(l, dst, width, stride, i);
            dst += stride;
        }
    } else {
        for (i = 0; i < height; i++) {
            lag_pred_line_yuy2(l, dst, width, stride, i,
                               width == l->avctx->width);
            dst += stride;
        }
    }

    return 0;
}

/**
 * Decode a frame.
 * @param avctx codec context
 * @param data output AVFrame
 * @param data_size size of output data or 0 if no picture is returned
 * @param avpkt input packet
 * @return number of consumed bytes on success or negative if decode fails
 */
static int lag_decode_frame(AVCodecContext *avctx,
                            void *data, int *got_frame, AVPacket *avpkt)
{
    const uint8_t *buf = avpkt->data;
    unsigned int buf_size = avpkt->size;
    LagarithContext *l = avctx->priv_data;
    ThreadFrame frame = { .f = data };
    AVFrame *const p  = data;
    uint8_t frametype;
    uint32_t offset_gu = 0, offset_bv = 0, offset_ry = 9;
    uint32_t offs[4];
    uint8_t *srcs[4], *dst;
    int i, j, planes = 3;
    int ret;

    p->key_frame = 1;

    frametype = buf[0];

    offset_gu = AV_RL32(buf + 1);
    offset_bv = AV_RL32(buf + 5);

    switch (frametype) {
    case FRAME_SOLID_RGBA:
        avctx->pix_fmt = AV_PIX_FMT_RGB32;
    case FRAME_SOLID_GRAY:
        if (frametype == FRAME_SOLID_GRAY)
            if (avctx->bits_per_coded_sample == 24) {
                avctx->pix_fmt = AV_PIX_FMT_RGB24;
            } else {
                avctx->pix_fmt = AV_PIX_FMT_0RGB32;
                planes = 4;
            }

        if ((ret = ff_thread_get_buffer(avctx, &frame, 0)) < 0)
            return ret;

        dst = p->data[0];
        if (frametype == FRAME_SOLID_RGBA) {
        for (j = 0; j < avctx->height; j++) {
            for (i = 0; i < avctx->width; i++)
                AV_WN32(dst + i * 4, offset_gu);
            dst += p->linesize[0];
        }
        } else {
            for (j = 0; j < avctx->height; j++) {
                memset(dst, buf[1], avctx->width * planes);
                dst += p->linesize[0];
            }
        }
        break;
    case FRAME_SOLID_COLOR:
        if (avctx->bits_per_coded_sample == 24) {
            avctx->pix_fmt = AV_PIX_FMT_RGB24;
        } else {
            avctx->pix_fmt = AV_PIX_FMT_RGB32;
            offset_gu |= 0xFFU << 24;
        }

        if ((ret = ff_thread_get_buffer(avctx, &frame,0)) < 0)
            return ret;

        dst = p->data[0];
        for (j = 0; j < avctx->height; j++) {
            for (i = 0; i < avctx->width; i++)
                if (avctx->bits_per_coded_sample == 24) {
                    AV_WB24(dst + i * 3, offset_gu);
                } else {
                    AV_WN32(dst + i * 4, offset_gu);
                }
            dst += p->linesize[0];
        }
        break;
    case FRAME_ARITH_RGBA:
        avctx->pix_fmt = AV_PIX_FMT_RGB32;
        planes = 4;
        offset_ry += 4;
        offs[3] = AV_RL32(buf + 9);
    case FRAME_ARITH_RGB24:
    case FRAME_U_RGB24:
        if (frametype == FRAME_ARITH_RGB24 || frametype == FRAME_U_RGB24)
            avctx->pix_fmt = AV_PIX_FMT_RGB24;

        if ((ret = ff_thread_get_buffer(avctx, &frame, 0)) < 0)
            return ret;

        offs[0] = offset_bv;
        offs[1] = offset_gu;
        offs[2] = offset_ry;

        l->rgb_stride = FFALIGN(avctx->width, 16);
        av_fast_malloc(&l->rgb_planes, &l->rgb_planes_allocated,
                       l->rgb_stride * avctx->height * planes + 1);
        if (!l->rgb_planes) {
            av_log(avctx, AV_LOG_ERROR, "cannot allocate temporary buffer\n");
            return AVERROR(ENOMEM);
        }
        for (i = 0; i < planes; i++)
            srcs[i] = l->rgb_planes + (i + 1) * l->rgb_stride * avctx->height - l->rgb_stride;
        for (i = 0; i < planes; i++)
            if (buf_size <= offs[i]) {
                av_log(avctx, AV_LOG_ERROR,
                        "Invalid frame offsets\n");
                return AVERROR_INVALIDDATA;
            }

        for (i = 0; i < planes; i++)
            lag_decode_arith_plane(l, srcs[i],
                                   avctx->width, avctx->height,
                                   -l->rgb_stride, buf + offs[i],
                                   buf_size - offs[i]);
        dst = p->data[0];
        for (i = 0; i < planes; i++)
            srcs[i] = l->rgb_planes + i * l->rgb_stride * avctx->height;
        for (j = 0; j < avctx->height; j++) {
            for (i = 0; i < avctx->width; i++) {
                uint8_t r, g, b, a;
                r = srcs[0][i];
                g = srcs[1][i];
                b = srcs[2][i];
                r += g;
                b += g;
                if (frametype == FRAME_ARITH_RGBA) {
                    a = srcs[3][i];
                    AV_WN32(dst + i * 4, MKBETAG(a, r, g, b));
                } else {
                    dst[i * 3 + 0] = r;
                    dst[i * 3 + 1] = g;
                    dst[i * 3 + 2] = b;
                }
            }
            dst += p->linesize[0];
            for (i = 0; i < planes; i++)
                srcs[i] += l->rgb_stride;
        }
        break;
    case FRAME_ARITH_YUY2:
