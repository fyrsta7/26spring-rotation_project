#else /* CONFIG_HARDCODED_TABLES */
static uint8_t band_start_tab[51];
static uint8_t bin_to_band_tab[253];
#endif

static inline int calc_lowcomp1(int a, int b0, int b1, int c)
{
    if ((b0 + 256) == b1) {
        a = c;
    } else if (b0 > b1) {
        a = FFMAX(a - 64, 0);
    }
    return a;
}

static inline int calc_lowcomp(int a, int b0, int b1, int bin)
{
    if (bin < 7) {
        return calc_lowcomp1(a, b0, b1, 384);
    } else if (bin < 20) {
        return calc_lowcomp1(a, b0, b1, 320);
    } else {
        return FFMAX(a - 128, 0);
    }
}
