        -7.0935347449210549190e+00,
        -1.5453977791786851041e-02,
        -2.5172644670688975051e-05,
        -3.0517226450451067446e-08,
        -2.6843448573468483278e-11,
        -1.5982226675653184646e-14,
        -5.2487866627945699800e-18,
    };
    static const double q1[] = {
        -2.2335582639474375245e+15,
         7.8858692566751002988e+12,
        -1.2207067397808979846e+10,
         1.0377081058062166144e+07,
        -4.8527560179962773045e+03,
         1.0L,
    };
    static const double p2[] = {
        -2.2210262233306573296e-04,
         1.3067392038106924055e-02,
        -4.4700805721174453923e-01,
         5.5674518371240761397e+00,
        -2.3517945679239481621e+01,
         3.1611322818701131207e+01,
        -9.6090021968656180000e+00,
    };
    static const double q2[] = {
        -5.5194330231005480228e-04,
         3.2547697594819615062e-02,
        -1.1151759188741312645e+00,
         1.3982595353892851542e+01,
        -6.0228002066743340583e+01,
         8.5539563258012929600e+01,
        -3.1446690275135491500e+01,
        1.0L,
    };
    double y, r, factor;
    if (x == 0)
        return 1.0;
    x = fabs(x);
    if (x <= 15) {
        y = x * x;
        return eval_poly(p1, FF_ARRAY_ELEMS(p1), y) / eval_poly(q1, FF_ARRAY_ELEMS(q1), y);
    }
    else {
        y = 1 / x - 1.0 / 15;
        r = eval_poly(p2, FF_ARRAY_ELEMS(p2), y) / eval_poly(q2, FF_ARRAY_ELEMS(q2), y);
        factor = exp(x) / sqrt(x);
        return factor * r;
    }
}

/**
 * builds a polyphase filterbank.
 * @param factor resampling factor
 * @param scale wanted sum of coefficients for each filter
 * @param filter_type  filter type
 * @param kaiser_beta  kaiser window beta
 * @return 0 on success, negative on error
 */
static int build_filter(ResampleContext *c, void *filter, double factor, int tap_count, int alloc, int phase_count, int scale,
                        int filter_type, double kaiser_beta){
    int ph, i;
    double x, y, w, t, s;
    double *tab = av_malloc_array(tap_count+1,  sizeof(*tab));
    double *sin_lut = av_malloc_array(phase_count / 2 + 1, sizeof(*sin_lut));
    const int center= (tap_count-1)/2;

    if (!tab || !sin_lut)
        goto fail;

    /* if upsampling, only need to interpolate, no filter */
    if (factor > 1.0)
        factor = 1.0;

    av_assert0(phase_count == 1 || phase_count % 2 == 0);

    if (factor == 1.0) {
        for (ph = 0; ph <= phase_count / 2; ph++)
            sin_lut[ph] = sin(M_PI * ph / phase_count);
    }
    for(ph = 0; ph <= phase_count / 2; ph++) {
        double norm = 0;
        s = sin_lut[ph];
        for(i=0;i<=tap_count;i++) {
            x = M_PI * ((double)(i - center) - (double)ph / phase_count) * factor;
            if (x == 0) y = 1.0;
            else if (factor == 1.0)
                y = s / x;
            else
                y = sin(x) / x;
            switch(filter_type){
            case SWR_FILTER_TYPE_CUBIC:{
                const float d= -0.5; //first order derivative = -0.5
                x = fabs(((double)(i - center) - (double)ph / phase_count) * factor);
                if(x<1.0) y= 1 - 3*x*x + 2*x*x*x + d*(            -x*x + x*x*x);
                else      y=                       d*(-4 + 8*x - 5*x*x + x*x*x);
                break;}
            case SWR_FILTER_TYPE_BLACKMAN_NUTTALL:
                w = 2.0*x / (factor*tap_count);
                t = -cos(w);
                y *= 0.3635819 - 0.4891775 * t + 0.1365995 * (2*t*t-1) - 0.0106411 * (4*t*t*t - 3*t);
                break;
            case SWR_FILTER_TYPE_KAISER:
                w = 2.0*x / (factor*tap_count*M_PI);
                y *= bessel(kaiser_beta*sqrt(FFMAX(1-w*w, 0)));
                break;
            default:
                av_assert0(0);
            }

            tab[i] = y;
            s = -s;
            if (i < tap_count)
                norm += y;
        }

        /* normalize so that an uniform color remains the same */
        switch(c->format){
        case AV_SAMPLE_FMT_S16P:
            for(i=0;i<tap_count;i++)
                ((int16_t*)filter)[ph * alloc + i] = av_clip(lrintf(tab[i] * scale / norm), INT16_MIN, INT16_MAX);
            if (tap_count % 2 == 0) {
                for (i = 0; i < tap_count; i++)
                    ((int16_t*)filter)[(phase_count-ph) * alloc + tap_count-1-i] = ((int16_t*)filter)[ph * alloc + i];
            }
            else {
                for (i = 1; i <= tap_count; i++)
                    ((int16_t*)filter)[(phase_count-ph) * alloc + tap_count-i] =
                        av_clip(lrintf(tab[i] * scale / (norm - tab[0] + tab[tap_count])), INT16_MIN, INT16_MAX);
            }
            break;
        case AV_SAMPLE_FMT_S32P:
            for(i=0;i<tap_count;i++)
                ((int32_t*)filter)[ph * alloc + i] = av_clipl_int32(llrint(tab[i] * scale / norm));
            if (tap_count % 2 == 0) {
                for (i = 0; i < tap_count; i++)
                    ((int32_t*)filter)[(phase_count-ph) * alloc + tap_count-1-i] = ((int32_t*)filter)[ph * alloc + i];
            }
            else {
                for (i = 1; i <= tap_count; i++)
                    ((int32_t*)filter)[(phase_count-ph) * alloc + tap_count-i] =
                        av_clipl_int32(llrint(tab[i] * scale / (norm - tab[0] + tab[tap_count])));
            }
            break;
        case AV_SAMPLE_FMT_FLTP:
            for(i=0;i<tap_count;i++)
                ((float*)filter)[ph * alloc + i] = tab[i] * scale / norm;
            if (tap_count % 2 == 0) {
                for (i = 0; i < tap_count; i++)
                    ((float*)filter)[(phase_count-ph) * alloc + tap_count-1-i] = ((float*)filter)[ph * alloc + i];
            }
            else {
                for (i = 1; i <= tap_count; i++)
