static int plot_spectrum_column(AVFilterLink *inlink, AVFrame *insamples)
{
    int ret;
    AVFilterContext *ctx = inlink->dst;
    AVFilterLink *outlink = ctx->outputs[0];
    ShowSpectrumContext *s = ctx->priv;
    AVFrame *outpicref = s->outpicref;
    const double w = s->win_scale;
    const float g = s->gain;
    int h = s->orientation == VERTICAL ? s->channel_height : s->channel_width;

    int ch, plane, x, y;

    /* fill a new spectrum column */
    /* initialize buffer for combining to black */
    clear_combine_buffer(s, s->orientation == VERTICAL ? outlink->h : outlink->w);

    for (ch = 0; ch < s->nb_display_channels; ch++) {
        float *magnitudes = s->magnitudes[ch];
        float yf, uf, vf;

        /* decide color range */
        switch (s->mode) {
        case COMBINED:
            // reduce range by channel count
            yf = 256.0f / s->nb_display_channels;
            switch (s->color_mode) {
            case RAINBOW:
            case MORELAND:
            case NEBULAE:
            case FIRE:
            case FIERY:
            case FRUIT:
            case INTENSITY:
                uf = yf;
                vf = yf;
                break;
            case CHANNEL:
                /* adjust saturation for mixed UV coloring */
                /* this factor is correct for infinite channels, an approximation otherwise */
                uf = yf * M_PI;
                vf = yf * M_PI;
                break;
            default:
                av_assert0(0);
            }
            break;
        case SEPARATE:
            // full range
            yf = 256.0f;
            uf = 256.0f;
            vf = 256.0f;
            break;
        default:
            av_assert0(0);
        }

        if (s->color_mode == CHANNEL) {
            if (s->nb_display_channels > 1) {
                uf *= 0.5 * sin((2 * M_PI * ch) / s->nb_display_channels);
                vf *= 0.5 * cos((2 * M_PI * ch) / s->nb_display_channels);
            } else {
                uf = 0.0f;
                vf = 0.0f;
            }
        }
        uf *= s->saturation;
        vf *= s->saturation;

        /* draw the channel */
        for (y = 0; y < h; y++) {
            int row = (s->mode == COMBINED) ? y : ch * h + y;
            float *out = &s->combine_buffer[3 * row];

            /* get magnitude */
            float a = g * w * magnitudes[y];

            /* apply scale */
            switch (s->scale) {
            case LINEAR:
                break;
            case SQRT:
                a = sqrt(a);
                break;
            case CBRT:
                a = cbrt(a);
                break;
            case FOURTHRT:
                a = pow(a, 0.25);
                break;
            case FIFTHRT:
                a = pow(a, 0.20);
                break;
            case LOG:
                a = 1 + log10(av_clipd(a * w, 1e-6, 1)) / 6; // zero = -120dBFS
                break;
            default:
                av_assert0(0);
            }

            pick_color(s, yf, uf, vf, a, out);
        }
    }

    av_frame_make_writable(s->outpicref);
    /* copy to output */
    if (s->orientation == VERTICAL) {
        if (s->sliding == SCROLL) {
            for (plane = 0; plane < 3; plane++) {
                for (y = 0; y < outlink->h; y++) {
                    uint8_t *p = outpicref->data[plane] +
                                 y * outpicref->linesize[plane];
                    memmove(p, p + 1, outlink->w - 1);
                }
            }
            s->xpos = outlink->w - 1;
        } else if (s->sliding == RSCROLL) {
            for (plane = 0; plane < 3; plane++) {
                for (y = 0; y < outlink->h; y++) {
                    uint8_t *p = outpicref->data[plane] +
                                 y * outpicref->linesize[plane];
                    memmove(p + 1, p, outlink->w - 1);
                }
            }
            s->xpos = 0;
        }
        for (plane = 0; plane < 3; plane++) {
            uint8_t *p = outpicref->data[plane] +
                         (outlink->h - 1) * outpicref->linesize[plane] +
                         s->xpos;
            for (y = 0; y < outlink->h; y++) {
                *p = lrintf(av_clipf(s->combine_buffer[3 * y + plane], 0, 255));
                p -= outpicref->linesize[plane];
            }
        }
    } else {
        if (s->sliding == SCROLL) {
            for (plane = 0; plane < 3; plane++) {
                for (y = 1; y < outlink->h; y++) {
                    memmove(outpicref->data[plane] + (y-1) * outpicref->linesize[plane],
                            outpicref->data[plane] + (y  ) * outpicref->linesize[plane],
                            outlink->w);
                }
            }
            s->xpos = outlink->h - 1;
        } else if (s->sliding == RSCROLL) {
            for (plane = 0; plane < 3; plane++) {
                for (y = outlink->h - 1; y >= 1; y--) {
                    memmove(outpicref->data[plane] + (y  ) * outpicref->linesize[plane],
                            outpicref->data[plane] + (y-1) * outpicref->linesize[plane],
                            outlink->w);
                }
            }
            s->xpos = 0;
        }
        for (plane = 0; plane < 3; plane++) {
            uint8_t *p = outpicref->data[plane] +
                         s->xpos * outpicref->linesize[plane];
            for (x = 0; x < outlink->w; x++) {
                *p = lrintf(av_clipf(s->combine_buffer[3 * x + plane], 0, 255));
                p++;
            }
        }
    }

    if (s->sliding != FULLFRAME || s->xpos == 0)
        outpicref->pts = insamples->pts;

    s->xpos++;
    if (s->orientation == VERTICAL && s->xpos >= outlink->w)
        s->xpos = 0;
    if (s->orientation == HORIZONTAL && s->xpos >= outlink->h)
        s->xpos = 0;
    if (!s->single_pic && (s->sliding != FULLFRAME || s->xpos == 0)) {
        ret = ff_filter_frame(outlink, av_frame_clone(s->outpicref));
        if (ret < 0)
            return ret;
    }

    return s->win_size;
}