    }
    return 0;
}

#define CLIP_SYMM(a, b) av_clip(a, -(b), b)
/**
 * weaker deblocking very similar to the one described in 4.4.2 of JVT-A003r1
 */
static inline void rv40_weak_loop_filter(uint8_t *src, const int step,
                                         const int filter_p1, const int filter_q1,
                                         const int alpha, const int beta,
                                         const int lim_p0q0,
                                         const int lim_q1, const int lim_p1,
                                         const int diff_p1p0, const int diff_q1q0,
                                         const int diff_p1p2, const int diff_q1q2)
{
    uint8_t *cm = ff_cropTbl + MAX_NEG_CROP;
    int t, u, diff;

    t = src[0*step] - src[-1*step];
    if(!t)
        return;
    u = (alpha * FFABS(t)) >> 7;
    if(u > 3 - (filter_p1 && filter_q1))
        return;

    t <<= 2;
    if(filter_p1 && filter_q1)
        t += src[-2*step] - src[1*step];
    diff = CLIP_SYMM((t + 4) >> 3, lim_p0q0);
    src[-1*step] = cm[src[-1*step] + diff];
    src[ 0*step] = cm[src[ 0*step] - diff];
    if(FFABS(diff_p1p2) <= beta && filter_p1){
        t = (diff_p1p0 + diff_p1p2 - diff) >> 1;
        src[-2*step] = cm[src[-2*step] - CLIP_SYMM(t, lim_p1)];
    }
    if(FFABS(diff_q1q2) <= beta && filter_q1){
        t = (diff_q1q0 + diff_q1q2 + diff) >> 1;
        src[ 1*step] = cm[src[ 1*step] - CLIP_SYMM(t, lim_q1)];
    }
}

static av_always_inline void rv40_adaptive_loop_filter(uint8_t *src, const int step,
                                             const int stride, const int dmode,
                                             const int lim_q1, const int lim_p1,
                                             const int alpha,
                                             const int beta, const int beta2,
                                             const int chroma, const int edge)
{
    int diff_p1p0[4], diff_q1q0[4], diff_p1p2[4], diff_q1q2[4];
    int sum_p1p0 = 0, sum_q1q0 = 0, sum_p1p2 = 0, sum_q1q2 = 0;
    uint8_t *ptr;
    int flag_strong0 = 1, flag_strong1 = 1;
    int filter_p1, filter_q1;
    int i;
    int lims;

    for(i = 0, ptr = src; i < 4; i++, ptr += stride){
        diff_p1p0[i] = ptr[-2*step] - ptr[-1*step];
        diff_q1q0[i] = ptr[ 1*step] - ptr[ 0*step];
        sum_p1p0 += diff_p1p0[i];
        sum_q1q0 += diff_q1q0[i];
    }
    filter_p1 = FFABS(sum_p1p0) < (beta<<2);
    filter_q1 = FFABS(sum_q1q0) < (beta<<2);
    if(!filter_p1 && !filter_q1)
        return;

    for(i = 0, ptr = src; i < 4; i++, ptr += stride){
        diff_p1p2[i] = ptr[-2*step] - ptr[-3*step];
        diff_q1q2[i] = ptr[ 1*step] - ptr[ 2*step];
        sum_p1p2 += diff_p1p2[i];
        sum_q1q2 += diff_q1q2[i];
    }

    if(edge){
        flag_strong0 = filter_p1 && (FFABS(sum_p1p2) < beta2);
        flag_strong1 = filter_q1 && (FFABS(sum_q1q2) < beta2);
    }else{
        flag_strong0 = flag_strong1 = 0;
    }

    lims = filter_p1 + filter_q1 + ((lim_q1 + lim_p1) >> 1) + 1;
    if(flag_strong0 && flag_strong1){ /* strong filtering */
        for(i = 0; i < 4; i++, src += stride){
            int sflag, p0, q0, p1, q1;
            int t = src[0*step] - src[-1*step];

            if(!t) continue;
            sflag = (alpha * FFABS(t)) >> 7;
