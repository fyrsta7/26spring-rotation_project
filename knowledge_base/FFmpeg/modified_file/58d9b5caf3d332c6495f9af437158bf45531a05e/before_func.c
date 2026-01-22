    for (int i = num-1; i > max; i--) {
        for (int j = 0; j < 2; j++) {
            int newlimit, sym;
            int t = j ? i-1 : i;
            int l = buf[t].bits;
            uint32_t code;

            sym = buf[t].symbol;
            if (l > curlimit)
                return;
            code = curcode + (buf[t].code >> curlen);
            newlimit = curlimit - l;
            l  += curlen;
            if (nb_elems>256) AV_WN16(info->val+2*curlevel, sym);
            else info->val[curlevel] = sym&0xFF;

            if (curlevel) { // let's not add single entries
                uint32_t val = code >> (32 - numbits);
                uint32_t  nb = val + (1U << (numbits - l));
                info->len = l;
                info->num = curlevel+1;
                for (; val < nb; val++)
                    AV_COPY64(table+val, info);
                levelcnt[curlevel-1]++;
            }

            if (curlevel+1 < VLC_MULTI_MAX_SYMBOLS && newlimit >= minlen) {
                add_level(table, nb_elems, num, numbits, buf,
                          code, l, newlimit, curlevel+1,
                          minlen, max, levelcnt, info);
            }
