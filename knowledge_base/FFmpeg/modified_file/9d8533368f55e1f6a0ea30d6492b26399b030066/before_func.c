        p += 2;
    } while (--j);

    /* pass 1 */

    p = &z[0];
    j = np >> 2;
    do {
        BF(p[0].re, p[0].im, p[2].re,  p[2].im,
           p[0].re, p[0].im, p[2].re,  p[2].im);
        BF(p[1].re, p[1].im, p[3].re,  p[3].im,
           p[1].re, p[1].im, p[3].im, -p[3].re);
        p+=4;
    } while (--j);

    /* pass 2 .. ln-1 */

    nblocks = np >> 3;
    nloops  =  1 << 2;
    np2     = np >> 1;
    do {
        p = z;
        q = z + nloops;
        for (j = 0; j < nblocks; j++) {
            BF(p->re, p->im, q->re, q->im,
               p->re, p->im, q->re, q->im);
            p++;
            q++;
            for(l = nblocks; l < np2; l += nblocks) {
                CMUL(tmp_re, tmp_im, costab[l], -sintab[l], q->re, q->im);
