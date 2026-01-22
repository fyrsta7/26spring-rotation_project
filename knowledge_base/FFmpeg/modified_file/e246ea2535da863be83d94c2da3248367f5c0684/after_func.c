            p = mp_get_yuv_from_rgb(mp, 0, y);
        } else {
            p.y += mp_gradient(mp, 0, mp_get_vlc(mp, gb));
            p.y = av_clip_uintp2(p.y, 5);
            if ((y & 3) == 0) {
                p.v += mp_gradient(mp, 1, mp_get_vlc(mp, gb));
                p.v = av_clip_intp2(p.v, 5);
                p.u += mp_gradient(mp, 2, mp_get_vlc(mp, gb));
                p.u = av_clip_intp2(p.u, 5);
            }
            mp->vpt[y] = p;
            mp_set_rgb_from_yuv(mp, 0, y, &p);
        }
    }
    for (y0 = 0; y0 < 2; ++y0)
        for (y = y0; y < mp->avctx->height; y += 2)
            mp_decode_line(mp, gb, y);
}

static int mp_decode_frame(AVCodecContext *avctx,
                                 void *data, int *got_frame,
                                 AVPacket *avpkt)
{
    const uint8_t *buf = avpkt->data;
    int buf_size = avpkt->size;
    MotionPixelsContext *mp = avctx->priv_data;
    GetBitContext gb;
    int i, count1, count2, sz, ret;

    if ((ret = ff_reget_buffer(avctx, mp->frame, 0)) < 0)
        return ret;

    /* le32 bitstream msb first */
    av_fast_padded_malloc(&mp->bswapbuf, &mp->bswapbuf_size, buf_size);
    if (!mp->bswapbuf)
        return AVERROR(ENOMEM);
    mp->bdsp.bswap_buf((uint32_t *) mp->bswapbuf, (const uint32_t *) buf,
                       buf_size / 4);
    if (buf_size & 3)
        memcpy(mp->bswapbuf + (buf_size & ~3), buf + (buf_size & ~3), buf_size & 3);
    init_get_bits(&gb, mp->bswapbuf, buf_size * 8);

    memset(mp->changes_map, 0, avctx->width * avctx->height);
    for (i = !(avctx->extradata[1] & 2); i < 2; ++i) {
        count1 = get_bits(&gb, 12);
        count2 = get_bits(&gb, 12);
        mp_read_changes_map(mp, &gb, count1, 8, i);
        mp_read_changes_map(mp, &gb, count2, 4, i);
    }

    mp->codes_count = get_bits(&gb, 4);
    if (mp->codes_count == 0)
        goto end;

    if (mp->changes_map[0] == 0) {
        *(uint16_t *)mp->frame->data[0] = get_bits(&gb, 15);
        mp->changes_map[0] = 1;
    }
    if (mp_read_codes_table(mp, &gb) < 0)
        goto end;
