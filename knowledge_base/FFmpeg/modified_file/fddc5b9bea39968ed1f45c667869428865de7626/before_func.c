    for (i = 0; i < len; i++) {
        if (fc_in[i]) {
            for (k = 0; k < i; k++)
                fc_out[k] += (fc_in[i] * filter[len + k - i]) >> 15;

            for (k = i; k < len; k++)
                fc_out[k] += (fc_in[i] * filter[      k - i]) >> 15;
        }
    }
}

void ff_celp_circ_addf(float *out, const float *in,
                       const float *lagged, int lag, float fac, int n)
{
    int k;
    for (k = 0; k < lag; k++)
        out[k] = in[k] + fac * lagged[n + k - lag];
    for (; k < n; k++)
        out[k] = in[k] + fac * lagged[    k - lag];
}

int ff_celp_lp_synthesis_filter(int16_t *out, const int16_t *filter_coeffs,
                                const int16_t *in, int buffer_length,
                                int filter_length, int stop_on_overflow,
