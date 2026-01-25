/**
 * Calculate a 512-point MDCT
 * @param out 256 output frequency coefficients
 * @param in  512 windowed input audio samples
 */
static void mdct512(int32_t *out, int16_t *in)
{
    int i, re, im, re1, im1;
    int16_t rot[MDCT_SAMPLES];
    IComplex x[MDCT_SAMPLES/4];

    /* shift to simplify computations */
    for (i = 0; i < MDCT_SAMPLES/4; i++)
        rot[i] = -in[i + 3*MDCT_SAMPLES/4];
    memcpy(&rot[MDCT_SAMPLES/4], &in[0], 3*MDCT_SAMPLES/4*sizeof(*in));

    /* pre rotation */
    for (i = 0; i < MDCT_SAMPLES/4; i++) {
        re =  ((int)rot[               2*i] - (int)rot[MDCT_SAMPLES  -1-2*i]) >> 1;
        im = -((int)rot[MDCT_SAMPLES/2+2*i] - (int)rot[MDCT_SAMPLES/2-1-2*i]) >> 1;
        CMUL(x[i].re, x[i].im, re, im, -xcos1[i], xsin1[i]);
    }

    fft(x, MDCT_NBITS - 2);

    /* post rotation */
    for (i = 0; i < MDCT_SAMPLES/4; i++) {
        re = x[i].re;
        im = x[i].im;
        CMUL(re1, im1, re, im, xsin1[i], xcos1[i]);
        out[                 2*i] = im1;
        out[MDCT_SAMPLES/2-1-2*i] = re1;
    }
}