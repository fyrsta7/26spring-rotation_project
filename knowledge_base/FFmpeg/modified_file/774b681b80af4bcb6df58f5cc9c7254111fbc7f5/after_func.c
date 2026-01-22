            coef = -coef;
        *dst = (coef*q + 8) >> 4;
    }
}

/**
 * Decode 2x2 subblock of coefficients.
 */
static inline void decode_subblock(DCTELEM *dst, int code, const int is_block2, GetBitContext *gb, VLC *vlc, int q)
{
    int flags = modulo_three_table[code];

    decode_coeff(    dst+0*4+0, (flags >> 6)    , 3, gb, vlc, q);
    if(is_block2){
        decode_coeff(dst+1*4+0, (flags >> 4) & 3, 2, gb, vlc, q);
        decode_coeff(dst+0*4+1, (flags >> 2) & 3, 2, gb, vlc, q);
    }else{
        decode_coeff(dst+0*4+1, (flags >> 4) & 3, 2, gb, vlc, q);
        decode_coeff(dst+1*4+0, (flags >> 2) & 3, 2, gb, vlc, q);
    }
    decode_coeff(    dst+1*4+1, (flags >> 0) & 3, 2, gb, vlc, q);
}

/**
 * Decode a single coefficient.
 */
static inline void decode_subblock1(DCTELEM *dst, int code, GetBitContext *gb, VLC *vlc, int q)
{
    int coeff = modulo_three_table[code] >> 6;
    decode_coeff(dst, coeff, 3, gb, vlc, q);
}

static inline void decode_subblock3(DCTELEM *dst, int code, GetBitContext *gb, VLC *vlc,
