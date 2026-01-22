
                        // Sum pixel with weight
                        if (diff < diff_max)
                        {
                            const int diffidx = diff * weight_fact_table;

                            //float weight = exp(-diff*weightFact);
                            const float weight = exptable[diffidx];

                            tmp_data[yc*dst_w + xc].weight_sum += weight;
                            tmp_data[yc*dst_w + xc].pixel_sum  += weight * compare[(yc+dy)*bw + xc + dx];
                        }

                        integral_ptr1++;
                        integral_ptr2++;
                    }
                }
            }
        }
    }

    // Copy edges
    for (int y = 0; y < dst_h; y++)
    {
        for (int x = 0; x < n_half; x++)
        {
            *(dst + y * dst_s + x)               = *(src + y * bw - x - 1);
            *(dst + y * dst_s - x + (dst_w - 1)) = *(src + y * bw + x + dst_w);
        }
    }
    for (int y = 0; y < n_half; y++)
    {
        memcpy(dst +           y*dst_s, src -     (y+1)*bw, dst_w);
        memcpy(dst + (dst_h-y-1)*dst_s, src + (y+dst_h)*bw, dst_w);
    }

    // Copy main image
    uint8_t result;
    for (int y = n_half; y < dst_h-n_half; y++)
    {
        for (int x = n_half; x < dst_w-n_half; x++)
        {
            result = (uint8_t)(tmp_data[y*dst_w + x].pixel_sum / tmp_data[y*dst_w + x].weight_sum);
            *(dst + y*dst_s + x) = result ? result : *(src + y*bw + x);
        }
    }

    free(tmp_data);
    free(integral_mem);

}

static int nlmeans_init(hb_filter_object_t *filter,
                           hb_filter_init_t *init)
{
    filter->private_data = calloc(sizeof(struct hb_filter_private_s), 1);
    hb_filter_private_t *pv = filter->private_data;
    NLMeansFunctions *functions = &pv->functions;

    functions->build_integral = build_integral_scalar;
#if defined(ARCH_X86)
    nlmeans_init_x86(functions);
#endif

    // Mark parameters unset
    for (int c = 0; c < 3; c++)
    {
        pv->strength[c]    = -1;
        pv->origin_tune[c] = -1;
        pv->patch_size[c]  = -1;
        pv->range[c]       = -1;
        pv->nframes[c]     = -1;
        pv->prefilter[c]   = -1;
    }
    pv->threads = -1;

    // Read user parameters
    if (filter->settings != NULL)
    {
        hb_dict_t * dict = filter->settings;
        hb_dict_extract_double(&pv->strength[0],    dict, "y-strength");
        hb_dict_extract_double(&pv->origin_tune[0], dict, "y-origin-tune");
        hb_dict_extract_int(&pv->patch_size[0],     dict, "y-patch-size");
        hb_dict_extract_int(&pv->range[0],          dict, "y-range");
        hb_dict_extract_int(&pv->nframes[0],        dict, "y-frame-count");
        hb_dict_extract_int(&pv->prefilter[0],      dict, "y-prefilter");

        hb_dict_extract_double(&pv->strength[1],    dict, "cb-strength");
        hb_dict_extract_double(&pv->origin_tune[1], dict, "cb-origin-tune");
        hb_dict_extract_int(&pv->patch_size[1],     dict, "cb-patch-size");
        hb_dict_extract_int(&pv->range[1],          dict, "cb-range");
        hb_dict_extract_int(&pv->nframes[1],        dict, "cb-frame-count");
        hb_dict_extract_int(&pv->prefilter[1],      dict, "cb-prefilter");

        hb_dict_extract_double(&pv->strength[2],    dict, "cr-strength");
        hb_dict_extract_double(&pv->origin_tune[2], dict, "cr-origin-tune");
        hb_dict_extract_int(&pv->patch_size[2],     dict, "cr-patch-size");
        hb_dict_extract_int(&pv->range[2],          dict, "cr-range");
        hb_dict_extract_int(&pv->nframes[2],        dict, "cr-frame-count");
        hb_dict_extract_int(&pv->prefilter[2],      dict, "cr-prefilter");

        hb_dict_extract_int(&pv->threads,           dict, "threads");
    }

    // Cascade values
    // Cr not set; inherit Cb. Cb not set; inherit Y. Y not set; defaults.
    for (int c = 1; c < 3; c++)
    {
        if (pv->strength[c]    == -1) { pv->strength[c]    = pv->strength[c-1]; }
        if (pv->origin_tune[c] == -1) { pv->origin_tune[c] = pv->origin_tune[c-1]; }
        if (pv->patch_size[c]  == -1) { pv->patch_size[c]  = pv->patch_size[c-1]; }
        if (pv->range[c]       == -1) { pv->range[c]       = pv->range[c-1]; }
        if (pv->nframes[c]     == -1) { pv->nframes[c]     = pv->nframes[c-1]; }
        if (pv->prefilter[c]   == -1) { pv->prefilter[c]   = pv->prefilter[c-1]; }
    }

    for (int c = 0; c < 3; c++)
    {
        // Replace unset values with defaults
        if (pv->strength[c]    == -1) { pv->strength[c]    = c ? NLMEANS_STRENGTH_LUMA_DEFAULT    : NLMEANS_STRENGTH_CHROMA_DEFAULT; }
        if (pv->origin_tune[c] == -1) { pv->origin_tune[c] = c ? NLMEANS_ORIGIN_TUNE_LUMA_DEFAULT : NLMEANS_ORIGIN_TUNE_CHROMA_DEFAULT; }
        if (pv->patch_size[c]  == -1) { pv->patch_size[c]  = c ? NLMEANS_PATCH_SIZE_LUMA_DEFAULT  : NLMEANS_PATCH_SIZE_CHROMA_DEFAULT; }
        if (pv->range[c]       == -1) { pv->range[c]       = c ? NLMEANS_RANGE_LUMA_DEFAULT       : NLMEANS_RANGE_CHROMA_DEFAULT; }
        if (pv->nframes[c]     == -1) { pv->nframes[c]     = c ? NLMEANS_FRAMES_LUMA_DEFAULT      : NLMEANS_FRAMES_CHROMA_DEFAULT; }
        if (pv->prefilter[c]   == -1) { pv->prefilter[c]   = c ? NLMEANS_PREFILTER_LUMA_DEFAULT   : NLMEANS_PREFILTER_CHROMA_DEFAULT; }

        // Sanitize
        if (pv->strength[c] < 0)        { pv->strength[c] = 0; }
        if (pv->origin_tune[c] < 0.01)  { pv->origin_tune[c] = 0.01; } // avoid black artifacts
        if (pv->origin_tune[c] > 1)     { pv->origin_tune[c] = 1; }
        if (pv->patch_size[c] % 2 == 0) { pv->patch_size[c]--; }
        if (pv->patch_size[c] < 1)      { pv->patch_size[c] = 1; }
        if (pv->range[c] % 2 == 0)      { pv->range[c]--; }
        if (pv->range[c] < 1)           { pv->range[c] = 1; }
        if (pv->nframes[c] < 1)         { pv->nframes[c] = 1; }
        if (pv->nframes[c] > NLMEANS_FRAMES_MAX) { pv->nframes[c] = NLMEANS_FRAMES_MAX; }
        if (pv->prefilter[c] < 0)       { pv->prefilter[c] = 0; }

        if (pv->max_frames < pv->nframes[c]) pv->max_frames = pv->nframes[c];

        // Precompute exponential table
        float *exptable = &pv->exptable[c][0];
        float *weight_fact_table = &pv->weight_fact_table[c];
        int   *diff_max = &pv->diff_max[c];
        const float weight_factor        = 1.0/pv->patch_size[c]/pv->patch_size[c] / (pv->strength[c] * pv->strength[c]);
        const float min_weight_in_table  = 0.0005;
        const float stretch              = NLMEANS_EXPSIZE / (-log(min_weight_in_table));
        *(weight_fact_table)             = weight_factor * stretch;
        *(diff_max)                      = NLMEANS_EXPSIZE / *(weight_fact_table);
        for (int i = 0; i < NLMEANS_EXPSIZE; i++)
        {
            exptable[i] = exp(-i/stretch);
        }
        exptable[NLMEANS_EXPSIZE-1] = 0;
    }

    // Threads
    if (pv->threads < 1) {
        pv->threads = hb_get_cpu_count();

        // Reduce internal thread count where we have many logical cores
        // Too many threads increases CPU cache pressure, reducing performance
