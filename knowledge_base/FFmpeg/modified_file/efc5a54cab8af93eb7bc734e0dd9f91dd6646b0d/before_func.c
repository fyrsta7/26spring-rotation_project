                }
            }
            if (tx == TX_4X4 && edges[mode].needs_topright) {
                if (have_top && have_right &&
                    n_px_need + n_px_need_tr <= n_px_have) {
                    memcpy(&(*a)[4], &top[4], 4);
                } else {
                    memset(&(*a)[4], (*a)[3], 4);
                }
            }
        }
    }
    if (edges[mode].needs_left) {
        if (have_left) {
            int n_px_need = 4 << tx, i, n_px_have = (((s->rows - row) << !p) - y) * 4;
            uint8_t *dst = x == 0 ? dst_edge : dst_inner;
            ptrdiff_t stride = x == 0 ? stride_edge : stride_inner;

            if (n_px_need <= n_px_have) {
                for (i = 0; i < n_px_need; i++)
                    l[i] = dst[i * stride - 1];
            } else {
                for (i = 0; i < n_px_have; i++)
                    l[i] = dst[i * stride - 1];
                memset(&l[i], l[i - 1], n_px_need - n_px_have);
            }
        } else {
            memset(l, 129, 4 << tx);
        }
    }

    return mode;
}

static void intra_recon(AVCodecContext *ctx, ptrdiff_t y_off, ptrdiff_t uv_off)
{
    VP9Context *s = ctx->priv_data;
    VP9Block *const b = &s->b;
    int row = b->row, col = b->col;
    int w4 = bwh_tab[1][b->bs][0] << 1, step1d = 1 << b->tx, n;
    int h4 = bwh_tab[1][b->bs][1] << 1, x, y, step = 1 << (b->tx * 2);
    int end_x = FFMIN(2 * (s->cols - col), w4);
    int end_y = FFMIN(2 * (s->rows - row), h4);
    int tx = 4 * s->lossless + b->tx, uvtx = b->uvtx + 4 * s->lossless;
    int uvstep1d = 1 << b->uvtx, p;
    uint8_t *dst = b->dst[0], *dst_r = s->f->data[0] + y_off;

    for (n = 0, y = 0; y < end_y; y += step1d) {
        uint8_t *ptr = dst, *ptr_r = dst_r;
        for (x = 0; x < end_x; x += step1d, ptr += 4 * step1d,
                               ptr_r += 4 * step1d, n += step) {
            int mode = b->mode[b->bs > BS_8x8 && b->tx == TX_4X4 ?
                               y * 2 + x : 0];
            LOCAL_ALIGNED_16(uint8_t, a_buf, [48]);
            uint8_t *a = &a_buf[16], l[32];
            enum TxfmType txtp = vp9_intra_txfm_type[mode];
            int eob = b->tx > TX_8X8 ? AV_RN16A(&s->eob[n]) : s->eob[n];

            mode = check_intra_mode(s, mode, &a, ptr_r, s->f->linesize[0],
                                    ptr, b->y_stride, l,
                                    col, x, w4, row, y, b->tx, 0);
            s->dsp.intra_pred[b->tx][mode](ptr, b->y_stride, l, a);
            if (eob)
                s->dsp.itxfm_add[tx][txtp](ptr, b->y_stride,
                                           s->block + 16 * n, eob);
        }
        dst_r += 4 * s->f->linesize[0] * step1d;
