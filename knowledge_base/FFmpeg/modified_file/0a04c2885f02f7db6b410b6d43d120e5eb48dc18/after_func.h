#define C_QUANT 0.4054f

static inline void abs_pow34_v(float *out, const float *in, const int size)
{
    int i;
    for (i = 0; i < size; i++) {
        float a = fabsf(in[i]);
        out[i] = sqrtf(a * sqrtf(a));
    }
}

static inline float pos_pow34(float a)
{
