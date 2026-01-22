    FMT_PAIR_FUNC(AV_SAMPLE_FMT_FLT, AV_SAMPLE_FMT_S16),
    FMT_PAIR_FUNC(AV_SAMPLE_FMT_DBL, AV_SAMPLE_FMT_S16),
    FMT_PAIR_FUNC(AV_SAMPLE_FMT_U8 , AV_SAMPLE_FMT_S32),
    FMT_PAIR_FUNC(AV_SAMPLE_FMT_S16, AV_SAMPLE_FMT_S32),
    FMT_PAIR_FUNC(AV_SAMPLE_FMT_S32, AV_SAMPLE_FMT_S32),
    FMT_PAIR_FUNC(AV_SAMPLE_FMT_FLT, AV_SAMPLE_FMT_S32),
    FMT_PAIR_FUNC(AV_SAMPLE_FMT_DBL, AV_SAMPLE_FMT_S32),
    FMT_PAIR_FUNC(AV_SAMPLE_FMT_U8 , AV_SAMPLE_FMT_FLT),
    FMT_PAIR_FUNC(AV_SAMPLE_FMT_S16, AV_SAMPLE_FMT_FLT),
    FMT_PAIR_FUNC(AV_SAMPLE_FMT_S32, AV_SAMPLE_FMT_FLT),
    FMT_PAIR_FUNC(AV_SAMPLE_FMT_FLT, AV_SAMPLE_FMT_FLT),
    FMT_PAIR_FUNC(AV_SAMPLE_FMT_DBL, AV_SAMPLE_FMT_FLT),
    FMT_PAIR_FUNC(AV_SAMPLE_FMT_U8 , AV_SAMPLE_FMT_DBL),
    FMT_PAIR_FUNC(AV_SAMPLE_FMT_S16, AV_SAMPLE_FMT_DBL),
    FMT_PAIR_FUNC(AV_SAMPLE_FMT_S32, AV_SAMPLE_FMT_DBL),
    FMT_PAIR_FUNC(AV_SAMPLE_FMT_FLT, AV_SAMPLE_FMT_DBL),
    FMT_PAIR_FUNC(AV_SAMPLE_FMT_DBL, AV_SAMPLE_FMT_DBL),
};

static void cpy(uint8_t **dst, const uint8_t **src, int len){
    memcpy(*dst, *src, len);
}

AudioConvert *swri_audio_convert_alloc(enum AVSampleFormat out_fmt,
                                       enum AVSampleFormat in_fmt,
                                       int channels, const int *ch_map,
                                       int flags)
{
    AudioConvert *ctx;
    conv_func_type *f = fmt_pair_to_conv_functions[av_get_packed_sample_fmt(out_fmt) + AV_SAMPLE_FMT_NB*av_get_packed_sample_fmt(in_fmt)];

    if (!f)
