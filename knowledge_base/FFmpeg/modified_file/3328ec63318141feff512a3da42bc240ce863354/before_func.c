    exp_ptr = exponents;
    gain = g->global_gain - 210;
    shift = g->scalefac_scale + 1;

    bstab = band_size_long[s->sample_rate_index];
    pretab = mpa_pretab[g->preflag];
    for(i=0;i<g->long_end;i++) {
        v0 = gain - ((g->scale_factors[i] + pretab[i]) << shift);
        len = bstab[i];
        for(j=len;j>0;j--)
            *exp_ptr++ = v0;
    }

    if (g->short_start < 13) {
        bstab = band_size_short[s->sample_rate_index];
        gains[0] = gain - (g->subblock_gain[0] << 3);
        gains[1] = gain - (g->subblock_gain[1] << 3);
        gains[2] = gain - (g->subblock_gain[2] << 3);
        k = g->long_end;
        for(i=g->short_start;i<13;i++) {
            len = bstab[i];
            for(l=0;l<3;l++) {
                v0 = gains[l] - (g->scale_factors[k++] << shift);
                for(j=len;j>0;j--)
                *exp_ptr++ = v0;
            }
        }
    }
}

/* handle n = 0 too */
static inline int get_bitsz(GetBitContext *s, int n)
{
    if (n == 0)
        return 0;
    else
        return get_bits(s, n);
}

static int huffman_decode(MPADecodeContext *s, GranuleDef *g,
                          int16_t *exponents, int end_pos)
{
    int s_index;
    int linbits, code, x, y, l, v, i, j, k, pos;
    GetBitContext last_gb;
    VLC *vlc;

    /* low frequencies (called big values) */
    s_index = 0;
    for(i=0;i<3;i++) {
        j = g->region_size[i];
        if (j == 0)
            continue;
        /* select vlc table */
        k = g->table_select[i];
        l = mpa_huff_data[k][0];
        linbits = mpa_huff_data[k][1];
        vlc = &huff_vlc[l];

        if(!l){
            memset(&g->sb_hybrid[s_index], 0, sizeof(*g->sb_hybrid)*j);
            s_index += 2*j;
            continue;
        }

        /* read huffcode and compute each couple */
        for(;j>0;j--) {
            if (get_bits_count(&s->gb) >= end_pos)
                break;
            y = get_vlc2(&s->gb, vlc->table, 8, 3);
            x = y >> 4;
            y = y & 0x0f;

            dprintf("region=%d n=%d x=%d y=%d exp=%d\n",
                    i, g->region_size[i] - j, x, y, exponents[s_index]);
            if (x) {
                if (x == 15)
                    x += get_bitsz(&s->gb, linbits);
                v = l3_unscale(x, exponents[s_index]);
                if (get_bits1(&s->gb))
                    v = -v;
            } else {
                v = 0;
            }
            g->sb_hybrid[s_index++] = v;
            if (y) {
                if (y == 15)
                    y += get_bitsz(&s->gb, linbits);
                v = l3_unscale(y, exponents[s_index]);
                if (get_bits1(&s->gb))
                    v = -v;
            } else {
                v = 0;
            }
