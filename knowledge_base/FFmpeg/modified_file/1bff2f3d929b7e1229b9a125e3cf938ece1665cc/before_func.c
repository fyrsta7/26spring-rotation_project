                pred_T[2] = dst[-p->linesize[0] + 4 * x + 2];

                r = get_vlc2(gb, s->vlc[0].table, s->vlc[0].bits, 2);
                g = get_vlc2(gb, s->vlc[1].table, s->vlc[1].bits, 2);
                b = get_vlc2(gb, s->vlc[1].table, s->vlc[1].bits, 2);

                dst[4 * x + 0] = pred_L[0] = (r + ((3 * (pred_T[0] + pred_L[0]) - 2 * pred_TL[0]) >> 2)) & 0xff;
                dst[4 * x + 1] = pred_L[1] = (r + g + ((3 * (pred_T[1] + pred_L[1]) - 2 * pred_TL[1]) >> 2)) & 0xff;
                dst[4 * x + 2] = pred_L[2] = (r + g + b + ((3 * (pred_T[2] + pred_L[2]) - 2 * pred_TL[2]) >> 2)) & 0xff;

                pred_TL[0] = pred_T[0];
                pred_TL[1] = pred_T[1];
                pred_TL[2] = pred_T[2];
            }
        }
        dst += p->linesize[0];
    }
}

static int build_vlc(VLC *vlc, const uint8_t *len, int count)
{
    uint32_t codes[1024];
