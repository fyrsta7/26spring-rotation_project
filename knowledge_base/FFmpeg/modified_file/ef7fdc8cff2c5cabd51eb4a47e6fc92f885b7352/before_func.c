void RENAME(swri_noise_shaping)(SwrContext *s, AudioData *dsts, const AudioData *srcs, const AudioData *noises, int count){
    int i, j, pos, ch;
    int taps  = s->dither.ns_taps;
    float S   = s->dither.ns_scale;
    float S_1 = s->dither.ns_scale_1;

    for (ch=0; ch<srcs->ch_count; ch++) {
        const float *noise = ((const float *)noises->ch[ch]) + s->dither.noise_pos;
        const DELEM *src = (const DELEM*)srcs->ch[ch];
        DELEM *dst = (DELEM*)dsts->ch[ch];
        float *ns_errors = s->dither.ns_errors[ch];
        pos  = s->dither.ns_pos;
        for (i=0; i<count; i++) {
            double d1, d = src[i]*S_1;
            for(j=0; j<taps; j++)
                d -= s->dither.ns_coeffs[j] * ns_errors[pos + j];
            pos = pos ? pos - 1 : taps - 1;
            d1 = rint(d + noise[i]);
            ns_errors[pos + taps] = ns_errors[pos] = d1 - d;
            d1 *= S;
            CLIP(d1);
            dst[i] = d1;
        }
    }

    s->dither.ns_pos = pos;
}
