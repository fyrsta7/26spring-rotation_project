    int i;

    /*
     * If we consider U and V as the components of a 2D vector then its angle
     * is the hue and the norm is the saturation
     */
    while (h--) {
        for (i = 0; i < w; i++) {
            /* Normalize the components from range [16;140] to [-112;112] */
            u = usrc[i] - 128;
            v = vsrc[i] - 128;
            /*
             * Apply the rotation of the vector : (c * u) - (s * v)
             *                                    (s * u) + (c * v)
             * De-normalize the components (without forgetting to scale 128
             * by << 16)
             * Finally scale back the result by >> 16
             */
            new_u = ((c * u) - (s * v) + (1 << 15) + (128 << 16)) >> 16;
            new_v = ((s * u) + (c * v) + (1 << 15) + (128 << 16)) >> 16;

            /* Prevent a potential overflow */
            udst[i] = av_clip_uint8_c(new_u);
            vdst[i] = av_clip_uint8_c(new_v);
        }

        usrc += src_linesize;
        vsrc += src_linesize;
        udst += dst_linesize;
        vdst += dst_linesize;
    }
}

#define TS2D(ts) ((ts) == AV_NOPTS_VALUE ? NAN : (double)(ts))
#define TS2T(ts, tb) ((ts) == AV_NOPTS_VALUE ? NAN : (double)(ts) * av_q2d(tb))

static int filter_frame(AVFilterLink *inlink, AVFrame *inpic)
{
    HueContext *hue = inlink->dst->priv;
    AVFilterLink *outlink = inlink->dst->outputs[0];
    AVFrame *outpic;
    int direct = 0;

    if (av_frame_is_writable(inpic)) {
        direct = 1;
        outpic = inpic;
    } else {
    outpic = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!outpic) {
        av_frame_free(&inpic);
        return AVERROR(ENOMEM);
    }
    av_frame_copy_props(outpic, inpic);
    }

    if (!hue->flat_syntax) {
        hue->var_values[VAR_T]   = TS2T(inpic->pts, inlink->time_base);
        hue->var_values[VAR_PTS] = TS2D(inpic->pts);

        if (hue->saturation_expr) {
            hue->saturation = av_expr_eval(hue->saturation_pexpr, hue->var_values, NULL);

            if (hue->saturation < SAT_MIN_VAL || hue->saturation > SAT_MAX_VAL) {
                hue->saturation = av_clip(hue->saturation, SAT_MIN_VAL, SAT_MAX_VAL);
                av_log(inlink->dst, AV_LOG_WARNING,
