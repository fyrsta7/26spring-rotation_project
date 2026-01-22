    }

    return 0;
}

/**
 * Bitexact implementation of sqrt(val/2).
 */
static int16_t square_root(int val)
{
    int16_t res = 0;
    int16_t exp = 0x4000;
    int i;

    for (i = 0; i < 14; i ++) {
        int res_exp = res + exp;
        if (val >= res_exp * res_exp << 1)
            res += exp;
        exp >>= 1;
    }
    return res;
