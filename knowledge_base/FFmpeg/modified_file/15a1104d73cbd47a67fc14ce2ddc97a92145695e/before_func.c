    };
    ret = ff_append_outpad(ctx, &pad);
    if (ret < 0)
        return ret;

    /* summary */
    av_log(ctx, AV_LOG_VERBOSE, "EBU +%d scale\n", ebur128->meter);

    return 0;
}

#define HIST_POS(power) (int)(((power) - ABS_THRES) * HIST_GRAIN)

/* loudness and power should be set such as loudness = -0.691 +
 * 10*log10(power), we just avoid doing that calculus two times */
static int gate_update(struct integrator *integ, double power,
                       double loudness, int gate_thres)
{
    int ipower;
    double relative_threshold;
    int gate_hist_pos;

    /* update powers histograms by incrementing current power count */
    ipower = av_clip(HIST_POS(loudness), 0, HIST_SIZE - 1);
    integ->histogram[ipower].count++;

    /* compute relative threshold and get its position in the histogram */
    integ->sum_kept_powers += power;
    integ->nb_kept_powers++;
    relative_threshold = integ->sum_kept_powers / integ->nb_kept_powers;
    if (!relative_threshold)
        relative_threshold = 1e-12;
    integ->rel_threshold = LOUDNESS(relative_threshold) + gate_thres;
    gate_hist_pos = av_clip(HIST_POS(integ->rel_threshold), 0, HIST_SIZE - 1);

    return gate_hist_pos;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *insamples)
{
    int i, ch, idx_insample;
    AVFilterContext *ctx = inlink->dst;
    EBUR128Context *ebur128 = ctx->priv;
    const int nb_channels = ebur128->nb_channels;
    const int nb_samples  = insamples->nb_samples;
    const double *samples = (double *)insamples->data[0];
    AVFrame *pic = ebur128->outpicref;

#if CONFIG_SWRESAMPLE
    if (ebur128->peak_mode & PEAK_MODE_TRUE_PEAKS) {
        const double *swr_samples = ebur128->swr_buf;
        int ret = swr_convert(ebur128->swr_ctx, (uint8_t**)&ebur128->swr_buf, 19200,
                              (const uint8_t **)insamples->data, nb_samples);
        if (ret < 0)
            return ret;
        for (ch = 0; ch < nb_channels; ch++)
            ebur128->true_peaks_per_frame[ch] = 0.0;
        for (idx_insample = 0; idx_insample < ret; idx_insample++) {
            for (ch = 0; ch < nb_channels; ch++) {
                ebur128->true_peaks[ch] = FFMAX(ebur128->true_peaks[ch], fabs(*swr_samples));
                ebur128->true_peaks_per_frame[ch] = FFMAX(ebur128->true_peaks_per_frame[ch],
                                                          fabs(*swr_samples));
                swr_samples++;
            }
        }
    }
#endif

    for (idx_insample = 0; idx_insample < nb_samples; idx_insample++) {
        const int bin_id_400  = ebur128->i400.cache_pos;
        const int bin_id_3000 = ebur128->i3000.cache_pos;

#define MOVE_TO_NEXT_CACHED_ENTRY(time) do {                \
    ebur128->i##time.cache_pos++;                           \
    if (ebur128->i##time.cache_pos ==                       \
        ebur128->i##time.cache_size) {                      \
        ebur128->i##time.filled    = 1;                     \
        ebur128->i##time.cache_pos = 0;                     \
    }                                                       \
} while (0)

        MOVE_TO_NEXT_CACHED_ENTRY(400);
        MOVE_TO_NEXT_CACHED_ENTRY(3000);

        for (ch = 0; ch < nb_channels; ch++) {
            double bin;

            if (ebur128->peak_mode & PEAK_MODE_SAMPLES_PEAKS)
                ebur128->sample_peaks[ch] = FFMAX(ebur128->sample_peaks[ch], fabs(*samples));

            ebur128->x[ch * 3] = *samples++; // set X[i]

            if (!ebur128->ch_weighting[ch])
                continue;

            /* Y[i] = X[i]*b0 + X[i-1]*b1 + X[i-2]*b2 - Y[i-1]*a1 - Y[i-2]*a2 */
#define FILTER(Y, X, NUM, DEN) do {                                             \
            double *dst = ebur128->Y + ch*3;                                    \
            double *src = ebur128->X + ch*3;                                    \
            dst[2] = dst[1];                                                    \
            dst[1] = dst[0];                                                    \
            dst[0] = src[0]*NUM[0] + src[1]*NUM[1] + src[2]*NUM[2]              \
                                   - dst[1]*DEN[1] - dst[2]*DEN[2];             \
} while (0)

            // TODO: merge both filters in one?
            FILTER(y, x, ebur128->pre_b, ebur128->pre_a);  // apply pre-filter
            ebur128->x[ch * 3 + 2] = ebur128->x[ch * 3 + 1];
            ebur128->x[ch * 3 + 1] = ebur128->x[ch * 3    ];
            FILTER(z, y, ebur128->rlb_b, ebur128->rlb_a);  // apply RLB-filter

            bin = ebur128->z[ch * 3] * ebur128->z[ch * 3];

            /* add the new value, and limit the sum to the cache size (400ms or 3s)
             * by removing the oldest one */
            ebur128->i400.sum [ch] = ebur128->i400.sum [ch] + bin - ebur128->i400.cache [ch][bin_id_400];
            ebur128->i3000.sum[ch] = ebur128->i3000.sum[ch] + bin - ebur128->i3000.cache[ch][bin_id_3000];

            /* override old cache entry with the new value */
            ebur128->i400.cache [ch][bin_id_400 ] = bin;
            ebur128->i3000.cache[ch][bin_id_3000] = bin;
        }

        /* For integrated loudness, gating blocks are 400ms long with 75%
         * overlap (see BS.1770-2 p5), so a re-computation is needed each 100ms
         * (4800 samples at 48kHz). */
        if (++ebur128->sample_count == inlink->sample_rate / 10) {
            double loudness_400, loudness_3000;
            double power_400 = 1e-12, power_3000 = 1e-12;
            AVFilterLink *outlink = ctx->outputs[0];
            const int64_t pts = insamples->pts +
                av_rescale_q(idx_insample, (AVRational){ 1, inlink->sample_rate },
                             outlink->time_base);

            ebur128->sample_count = 0;

#define COMPUTE_LOUDNESS(m, time) do {                                              \
    if (ebur128->i##time.filled) {                                                  \
        /* weighting sum of the last <time> ms */                                   \
        for (ch = 0; ch < nb_channels; ch++)                                        \
            power_##time += ebur128->ch_weighting[ch] * ebur128->i##time.sum[ch];   \
        power_##time /= I##time##_BINS(inlink->sample_rate);                        \
    }                                                                               \
    loudness_##time = LOUDNESS(power_##time);                                       \
} while (0)

            COMPUTE_LOUDNESS(M,  400);
            COMPUTE_LOUDNESS(S, 3000);

            /* Integrated loudness */
#define I_GATE_THRES -10  // initially defined to -8 LU in the first EBU standard

            if (loudness_400 >= ABS_THRES) {
                double integrated_sum = 0.0;
                uint64_t nb_integrated = 0;
                int gate_hist_pos = gate_update(&ebur128->i400, power_400,
                                                loudness_400, I_GATE_THRES);

                /* compute integrated loudness by summing the histogram values
                 * above the relative threshold */
                for (i = gate_hist_pos; i < HIST_SIZE; i++) {
                    const unsigned nb_v = ebur128->i400.histogram[i].count;
                    nb_integrated  += nb_v;
                    integrated_sum += nb_v * ebur128->i400.histogram[i].energy;
                }
                if (nb_integrated) {
                    ebur128->integrated_loudness = LOUDNESS(integrated_sum / nb_integrated);
                    /* dual-mono correction */
                    if (nb_channels == 1 && ebur128->dual_mono) {
                        ebur128->integrated_loudness -= ebur128->pan_law;
                    }
                }
            }

            /* LRA */
#define LRA_GATE_THRES -20
#define LRA_LOWER_PRC   10
#define LRA_HIGHER_PRC  95

            /* XXX: example code in EBU 3342 is ">=" but formula in BS.1770
             * specs is ">" */
            if (loudness_3000 >= ABS_THRES) {
                uint64_t nb_powers = 0;
                int gate_hist_pos = gate_update(&ebur128->i3000, power_3000,
                                                loudness_3000, LRA_GATE_THRES);

                for (i = gate_hist_pos; i < HIST_SIZE; i++)
                    nb_powers += ebur128->i3000.histogram[i].count;
                if (nb_powers) {
                    uint64_t n, nb_pow;

                    /* get lower loudness to consider */
                    n = 0;
                    nb_pow = LRA_LOWER_PRC  * nb_powers / 100. + 0.5;
                    for (i = gate_hist_pos; i < HIST_SIZE; i++) {
                        n += ebur128->i3000.histogram[i].count;
                        if (n >= nb_pow) {
                            ebur128->lra_low = ebur128->i3000.histogram[i].loudness;
                            break;
                        }
                    }

                    /* get higher loudness to consider */
                    n = nb_powers;
                    nb_pow = LRA_HIGHER_PRC * nb_powers / 100. + 0.5;
                    for (i = HIST_SIZE - 1; i >= 0; i--) {
                        n -= ebur128->i3000.histogram[i].count;
                        if (n < nb_pow) {
                            ebur128->lra_high = ebur128->i3000.histogram[i].loudness;
                            break;
                        }
                    }

                    // XXX: show low & high on the graph?
                    ebur128->loudness_range = ebur128->lra_high - ebur128->lra_low;
                }
            }

            /* dual-mono correction */
            if (nb_channels == 1 && ebur128->dual_mono) {
                loudness_400 -= ebur128->pan_law;
                loudness_3000 -= ebur128->pan_law;
            }

#define LOG_FMT "TARGET:%d LUFS    M:%6.1f S:%6.1f     I:%6.1f %s       LRA:%6.1f LU"

            /* push one video frame */
            if (ebur128->do_video) {
                AVFrame *clone;
                int x, y, ret;
                uint8_t *p;
                double gauge_value;
                int y_loudness_lu_graph, y_loudness_lu_gauge;

                if (ebur128->gauge_type == GAUGE_TYPE_MOMENTARY) {
                    gauge_value = loudness_400 - ebur128->target;
                } else {
                    gauge_value = loudness_3000 - ebur128->target;
                }

                y_loudness_lu_graph = lu_to_y(ebur128, loudness_3000 - ebur128->target);
                y_loudness_lu_gauge = lu_to_y(ebur128, gauge_value);

                /* draw the graph using the short-term loudness */
                p = pic->data[0] + ebur128->graph.y*pic->linesize[0] + ebur128->graph.x*3;
                for (y = 0; y < ebur128->graph.h; y++) {
                    const uint8_t *c = get_graph_color(ebur128, y_loudness_lu_graph, y);

                    memmove(p, p + 3, (ebur128->graph.w - 1) * 3);
                    memcpy(p + (ebur128->graph.w - 1) * 3, c, 3);
                    p += pic->linesize[0];
                }

                /* draw the gauge using either momentary or short-term loudness */
                p = pic->data[0] + ebur128->gauge.y*pic->linesize[0] + ebur128->gauge.x*3;
                for (y = 0; y < ebur128->gauge.h; y++) {
                    const uint8_t *c = get_graph_color(ebur128, y_loudness_lu_gauge, y);

                    for (x = 0; x < ebur128->gauge.w; x++)
                        memcpy(p + x*3, c, 3);
                    p += pic->linesize[0];
                }

                /* draw textual info */
                if (ebur128->scale == SCALE_TYPE_ABSOLUTE) {
                    drawtext(pic, PAD, PAD - PAD/2, FONT16, font_colors,
                             LOG_FMT "     ", // padding to erase trailing characters
                             ebur128->target, loudness_400, loudness_3000,
                             ebur128->integrated_loudness, "LUFS", ebur128->loudness_range);
                } else {
                    drawtext(pic, PAD, PAD - PAD/2, FONT16, font_colors,
                             LOG_FMT "     ", // padding to erase trailing characters
                             ebur128->target, loudness_400-ebur128->target, loudness_3000-ebur128->target,
                             ebur128->integrated_loudness-ebur128->target, "LU", ebur128->loudness_range);
                }

                /* set pts and push frame */
                pic->pts = pts;
                clone = av_frame_clone(pic);
                if (!clone)
                    return AVERROR(ENOMEM);
                ret = ff_filter_frame(outlink, clone);
                if (ret < 0)
                    return ret;
            }

            if (ebur128->metadata) { /* happens only once per filter_frame call */
                char metabuf[128];
#define META_PREFIX "lavfi.r128."

#define SET_META(name, var) do {                                            \
    snprintf(metabuf, sizeof(metabuf), "%.3f", var);                        \
    av_dict_set(&insamples->metadata, name, metabuf, 0);                    \
} while (0)

#define SET_META_PEAK(name, ptype) do {                                     \
    if (ebur128->peak_mode & PEAK_MODE_ ## ptype ## _PEAKS) {               \
        char key[64];                                                       \
        for (ch = 0; ch < nb_channels; ch++) {                              \
            snprintf(key, sizeof(key),                                      \
                     META_PREFIX AV_STRINGIFY(name) "_peaks_ch%d", ch);     \
            SET_META(key, ebur128->name##_peaks[ch]);                       \
        }                                                                   \
    }                                                                       \
