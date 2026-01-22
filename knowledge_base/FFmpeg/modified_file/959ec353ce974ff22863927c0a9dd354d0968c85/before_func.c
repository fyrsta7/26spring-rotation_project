                                      startx_safe, safe_ph);

    // main and safe part of the integral
    av_assert1(startx_safe - s1x >= 0); av_assert1(startx_safe - s1x < w);
    av_assert1(starty_safe - s1y >= 0); av_assert1(starty_safe - s1y < h);
    av_assert1(startx_safe - s2x >= 0); av_assert1(startx_safe - s2x < w);
    av_assert1(starty_safe - s2y >= 0); av_assert1(starty_safe - s2y < h);
    if (safe_pw && safe_ph)
        dsp->compute_safe_ssd_integral_image(ii + starty_safe*ii_linesize_32 + startx_safe, ii_linesize_32,
                                             src + (starty_safe - s1y) * linesize + (startx_safe - s1x), linesize,
                                             src + (starty_safe - s2y) * linesize + (startx_safe - s2x), linesize,
                                             safe_pw, safe_ph);

    // right part of the integral
    compute_unsafe_ssd_integral_image(ii, ii_linesize_32,
                                      endx_safe, starty_safe,
                                      src, linesize,
                                      offx, offy, e, w, h,
                                      ii_w - endx_safe, safe_ph);

    // bottom part where only one of s1 and s2 is still readable, or none at all
    compute_unsafe_ssd_integral_image(ii, ii_linesize_32,
                                      0, endy_safe,
                                      src, linesize,
                                      offx, offy, e, w, h,
                                      ii_w, ii_h - endy_safe);
}

static int config_input(AVFilterLink *inlink)
{
    AVFilterContext *ctx = inlink->dst;
    NLMeansContext *s = ctx->priv;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);
    const int e = FFMAX(s->research_hsize, s->research_hsize_uv)
                + FFMAX(s->patch_hsize,    s->patch_hsize_uv);

    s->chroma_w = AV_CEIL_RSHIFT(inlink->w, desc->log2_chroma_w);
