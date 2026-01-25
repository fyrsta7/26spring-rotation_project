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
            inrow  = inrow0;
            outrow = outrow0;
            for (j = 0; j < inlink->w; j++) {
                for (k = 0; k < lut->step; k++)
                    outrow[k] = lut->lut[lut->rgba_map[k]][inrow[k]];
                outrow += lut->step;
                inrow  += lut->step;
            }
            inrow0  += inpic ->linesize[0];
            outrow0 += outpic->linesize[0];
        }
    } else {
        /* planar */
        for (plane = 0; plane < 4 && inpic->data[plane]; plane++) {
            int vsub = plane == 1 || plane == 2 ? lut->vsub : 0;
            int hsub = plane == 1 || plane == 2 ? lut->hsub : 0;

            inrow  = inpic ->data[plane] + (y>>vsub) * inpic ->linesize[plane];
            outrow = outpic->data[plane] + (y>>vsub) * outpic->linesize[plane];

            for (i = 0; i < h>>vsub; i ++) {
                const uint8_t *tab = lut->lut[plane];
                int w = inlink->w>>hsub;
                for (j = 0; j < w; j++)
                    outrow[j] = tab[inrow[j]];
                inrow  += inpic ->linesize[plane];
                outrow += outpic->linesize[plane];
            }
        }
    }

    avfilter_draw_slice(outlink, y, h, slice_dir);
}