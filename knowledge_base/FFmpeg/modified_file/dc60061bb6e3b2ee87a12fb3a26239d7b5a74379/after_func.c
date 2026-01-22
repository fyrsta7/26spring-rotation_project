                           bpass_csty_symbol && (clnpass_cnt >= 4), vert_causal_ctx_csty_symbol);
            break;
        case 1:
            decode_refpass(t1, width, height, bpno + 1);
            if (bpass_csty_symbol && clnpass_cnt >= 4)
                ff_mqc_initdec(&t1->mqc, cblk->data);
            break;
        case 2:
            decode_clnpass(s, t1, width, height, bpno + 1, bandpos,
                           codsty->cblk_style & JPEG2000_CBLK_SEGSYM, vert_causal_ctx_csty_symbol);
            clnpass_cnt = clnpass_cnt + 1;
            if (bpass_csty_symbol && clnpass_cnt >= 4)
                ff_mqc_initdec(&t1->mqc, cblk->data);
