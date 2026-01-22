
    val   = (k0 - K + s) ^ s;
    norm += val * val;
    *y++  = val;

    /* N==1 */
    s     = -i;
    val   = (K + s) ^ s;
    norm += val * val;
    *y    = val;

    return norm;
}

static inline void celt_encode_pulses(OpusRangeCoder *rc, int *y, uint32_t N, uint32_t K)
{
    ff_opus_rc_enc_uint(rc, celt_icwrsi(N, K, y), CELT_PVQ_V(N, K));
}

static inline float celt_decode_pulses(OpusRangeCoder *rc, int *y, uint32_t N, uint32_t K)
{
    const uint32_t idx = ff_opus_rc_dec_uint(rc, CELT_PVQ_V(N, K));
    return celt_cwrsi(N, K, idx, y);
}

/*
 * Faster than libopus's search, operates entirely in the signed domain.
 * Slightly worse/better depending on N, K and the input vector.
 */
static float ppp_pvq_search_c(float *X, int *y, int K, int N)
{
    int i, y_norm = 0;
    float res = 0.0f, xy_norm = 0.0f;

    for (i = 0; i < N; i++)
        res += FFABS(X[i]);

    res = K/(res + FLT_EPSILON);

    for (i = 0; i < N; i++) {
        y[i] = lrintf(res*X[i]);
        y_norm  += y[i]*y[i];
        xy_norm += y[i]*X[i];
        K -= FFABS(y[i]);
    }

