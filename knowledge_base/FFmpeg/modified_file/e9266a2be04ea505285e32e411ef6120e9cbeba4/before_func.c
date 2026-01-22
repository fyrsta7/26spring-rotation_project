        /* Y */
        for (y = 0; y < 4; y++) {
            for (x = 0; x < 4; x++) {
                vp8_mc_luma(s, dst[0] + 4*y*s->linesize + x*4,
                            ref->data[0], &bmv[4*y + x],
                            4*x + x_off, 4*y + y_off, 4, 4,
                            width, height, s->linesize,
                            s->put_pixels_tab[2]);
            }
        }

        /* U/V */
        x_off >>= 1; y_off >>= 1; width >>= 1; height >>= 1;
        for (y = 0; y < 2; y++) {
            for (x = 0; x < 2; x++) {
                uvmv.x = mb->bmv[ 2*y    * 4 + 2*x  ].x +
                         mb->bmv[ 2*y    * 4 + 2*x+1].x +
                         mb->bmv[(2*y+1) * 4 + 2*x  ].x +
                         mb->bmv[(2*y+1) * 4 + 2*x+1].x;
                uvmv.y = mb->bmv[ 2*y    * 4 + 2*x  ].y +
                         mb->bmv[ 2*y    * 4 + 2*x+1].y +
                         mb->bmv[(2*y+1) * 4 + 2*x  ].y +
                         mb->bmv[(2*y+1) * 4 + 2*x+1].y;
                uvmv.x = (uvmv.x + 2 + (uvmv.x >> (INT_BIT-1))) >> 2;
                uvmv.y = (uvmv.y + 2 + (uvmv.y >> (INT_BIT-1))) >> 2;
                if (s->profile == 3) {
                    uvmv.x &= ~7;
                    uvmv.y &= ~7;
                }
                vp8_mc_chroma(s, dst[1] + 4*y*s->uvlinesize + x*4,
                              dst[2] + 4*y*s->uvlinesize + x*4,
