
        /* compute the lut */
        lut->var_values[VAR_MAXVAL] = max[comp];
        lut->var_values[VAR_MINVAL] = min[comp];

        for (val = 0; val < 256; val++) {
            lut->var_values[VAR_VAL] = val;
            lut->var_values[VAR_CLIPVAL] = av_clip(val, min[comp], max[comp]);
            lut->var_values[VAR_NEGVAL] =
                av_clip(min[comp] + max[comp] - lut->var_values[VAR_VAL],
                        min[comp], max[comp]);

            res = av_expr_eval(lut->comp_expr[comp], lut->var_values, lut);
            if (isnan(res)) {
                av_log(ctx, AV_LOG_ERROR,
                       "Error when evaluating the expression '%s' for the value %d for the component #%d.\n",
                       lut->comp_expr_str[comp], val, comp);
                return AVERROR(EINVAL);
            }
            lut->lut[comp][val] = av_clip((int)res, min[comp], max[comp]);
            av_log(ctx, AV_LOG_DEBUG, "val[%d][%d] = %d\n", comp, val, lut->lut[comp][val]);
        }
    }

    return 0;
}

static void draw_slice(AVFilterLink *inlink, int y, int h, int slice_dir)
{
    AVFilterContext *ctx = inlink->dst;
    LutContext *lut = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    AVFilterBufferRef *inpic  = inlink ->cur_buf;
    AVFilterBufferRef *outpic = outlink->out_buf;
    uint8_t *inrow, *outrow, *inrow0, *outrow0;
    int i, j, k, plane;

    if (lut->is_rgb) {
        /* packed */
        inrow0  = inpic ->data[0] + y * inpic ->linesize[0];
        outrow0 = outpic->data[0] + y * outpic->linesize[0];

        for (i = 0; i < h; i ++) {
            int w = inlink->w;
            inrow  = inrow0;
            outrow = outrow0;
            for (j = 0; j < w; j++) {
                outrow[0] = lut->lut[lut->rgba_map[0]][inrow[0]];
                if (lut->step>1) {
                    outrow[1] = lut->lut[lut->rgba_map[1]][inrow[1]];
                    if (lut->step>2) {
                        outrow[2] = lut->lut[lut->rgba_map[2]][inrow[2]];
                        if (lut->step>3) {
                            outrow[3] = lut->lut[lut->rgba_map[3]][inrow[3]];
                        }
                    }
                }
                outrow += lut->step;
