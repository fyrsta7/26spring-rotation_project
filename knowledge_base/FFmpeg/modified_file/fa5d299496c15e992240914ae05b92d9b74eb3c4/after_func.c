        { "ring2",       "", 0, AV_OPT_TYPE_CONST, {.i64=TEST_RING2},       INT_MIN, INT_MAX, FLAGS, "test" },
        { "all",         "", 0, AV_OPT_TYPE_CONST, {.i64=TEST_ALL},         INT_MIN, INT_MAX, FLAGS, "test" },
    { NULL }
};

AVFILTER_DEFINE_CLASS(mptestsrc);

static double c[64];

static void init_idct(void)
{
    int i, j;

    for (i = 0; i < 8; i++) {
        double s = i == 0 ? sqrt(0.125) : 0.5;

        for (j = 0; j < 8; j++)
            c[i*8+j] = s*cos((M_PI/8.0)*i*(j+0.5));
    }
}

static void idct(uint8_t *dst, int dst_linesize, int src[64])
{
    int i, j, k;
    double tmp[64];

    for (i = 0; i < 8; i++) {
