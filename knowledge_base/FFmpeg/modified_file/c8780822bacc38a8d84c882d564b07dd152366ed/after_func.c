 1.0/(21*21), 1.0/(22*22), 1.0/(23*23), 1.0/(24*24), 1.0/(25*25), 1.0/(26*26), 1.0/(27*27), 1.0/(28*28), 1.0/(29*29), 1.0/(30*30),
 1.0/(31*31), 1.0/(32*32), 1.0/(33*33), 1.0/(34*34), 1.0/(35*35), 1.0/(36*36), 1.0/(37*37), 1.0/(38*38), 1.0/(39*39), 1.0/(40*40),
 1.0/(41*41), 1.0/(42*42), 1.0/(43*43), 1.0/(44*44), 1.0/(45*45), 1.0/(46*46), 1.0/(47*47), 1.0/(48*48), 1.0/(49*49), 1.0/(50*50),
 1.0/(51*51), 1.0/(52*52), 1.0/(53*53), 1.0/(54*54), 1.0/(55*55), 1.0/(56*56), 1.0/(57*57), 1.0/(58*58), 1.0/(59*59), 1.0/(60*60),
 1.0/(61*61), 1.0/(62*62), 1.0/(63*63), 1.0/(64*64), 1.0/(65*65), 1.0/(66*66), 1.0/(67*67), 1.0/(68*68), 1.0/(69*69), 1.0/(70*70),
 1.0/(71*71), 1.0/(72*72), 1.0/(73*73), 1.0/(74*74), 1.0/(75*75), 1.0/(76*76), 1.0/(77*77), 1.0/(78*78), 1.0/(79*79), 1.0/(80*80),
 1.0/(81*81), 1.0/(82*82), 1.0/(83*83), 1.0/(84*84), 1.0/(85*85), 1.0/(86*86), 1.0/(87*87), 1.0/(88*88), 1.0/(89*89), 1.0/(90*90),
 1.0/(91*91), 1.0/(92*92), 1.0/(93*93), 1.0/(94*94), 1.0/(95*95), 1.0/(96*96), 1.0/(97*97), 1.0/(98*98), 1.0/(99*99), 1.0/(10000)
    };

    x= x*x/4;
    t = x;
    v = 1 + x;
    for(i=1; v != lastv; i+=2){
        t *= x*inv[i];
        v += t;
        lastv=v;
        t *= x*inv[i + 1];
        v += t;
        av_assert2(i<98);
    }
    return v;
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
                        int filter_type, int kaiser_beta){
    int ph, i;
    double x, y, w, t;
    double *tab = av_malloc_array(tap_count+1,  sizeof(*tab));
    const int center= (tap_count-1)/2;

    if (!tab)
        return AVERROR(ENOMEM);

    /* if upsampling, only need to interpolate, no filter */
    if (factor > 1.0)
        factor = 1.0;

    av_assert0(phase_count == 1 || phase_count % 2 == 0);
    for(ph = 0; ph <= phase_count / 2; ph++) {
        double norm = 0;
        for(i=0;i<=tap_count;i++) {
            x = M_PI * ((double)(i - center) - (double)ph / phase_count) * factor;
            if (x == 0) y = 1.0;
            else        y = sin(x) / x;
            switch(filter_type){
            case SWR_FILTER_TYPE_CUBIC:{
                const float d= -0.5; //first order derivative = -0.5
                x = fabs(((double)(i - center) - (double)ph / phase_count) * factor);
                if(x<1.0) y= 1 - 3*x*x + 2*x*x*x + d*(            -x*x + x*x*x);
                else      y=                       d*(-4 + 8*x - 5*x*x + x*x*x);
                break;}
            case SWR_FILTER_TYPE_BLACKMAN_NUTTALL:
                w = 2.0*x / (factor*tap_count) + M_PI;
                t = cos(w);
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
                    ((float*)filter)[(phase_count-ph) * alloc + tap_count-i] = tab[i] * scale / (norm - tab[0] + tab[tap_count]);
            }
            break;
        case AV_SAMPLE_FMT_DBLP:
            for(i=0;i<tap_count;i++)
                ((double*)filter)[ph * alloc + i] = tab[i] * scale / norm;
            if (tap_count % 2 == 0) {
                for (i = 0; i < tap_count; i++)
                    ((double*)filter)[(phase_count-ph) * alloc + tap_count-1-i] = ((double*)filter)[ph * alloc + i];
            }
            else {
                for (i = 1; i <= tap_count; i++)
                    ((double*)filter)[(phase_count-ph) * alloc + tap_count-i] = tab[i] * scale / (norm - tab[0] + tab[tap_count]);
            }
            break;
        }
    }
#if 0
    {
#define LEN 1024
        int j,k;
        double sine[LEN + tap_count];
        double filtered[LEN];
        double maxff=-2, minff=2, maxsf=-2, minsf=2;
        for(i=0; i<LEN; i++){
