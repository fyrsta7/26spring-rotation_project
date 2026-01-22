static const uint8_t quant_lut_offset[8] = { 0, 0, 1, 4, 11, 32, 81, 230 };

static void apply_mdct(NellyMoserEncodeContext *s)
{
    float *in0 = s->buf;
    float *in1 = s->buf + NELLY_BUF_LEN;
    float *in2 = s->buf + 2 * NELLY_BUF_LEN;

    s->fdsp->vector_fmul        (s->in_buff,                 in0, ff_sine_128, NELLY_BUF_LEN);
    s->fdsp->vector_fmul_reverse(s->in_buff + NELLY_BUF_LEN, in1, ff_sine_128, NELLY_BUF_LEN);
    s->mdct_ctx.mdct_calc(&s->mdct_ctx, s->mdct_out, s->in_buff);

    s->fdsp->vector_fmul        (s->in_buff,                 in1, ff_sine_128, NELLY_BUF_LEN);
    s->fdsp->vector_fmul_reverse(s->in_buff + NELLY_BUF_LEN, in2, ff_sine_128, NELLY_BUF_LEN);
    s->mdct_ctx.mdct_calc(&s->mdct_ctx, s->mdct_out + NELLY_BUF_LEN, s->in_buff);
}

static av_cold int encode_end(AVCodecContext *avctx)
{
    NellyMoserEncodeContext *s = avctx->priv_data;

    ff_mdct_end(&s->mdct_ctx);

    if (s->avctx->trellis) {
        av_freep(&s->opt);
        av_freep(&s->path);
    }
    ff_af_queue_close(&s->afq);
    av_freep(&s->fdsp);

    return 0;
}

static av_cold int encode_init(AVCodecContext *avctx)
{
    NellyMoserEncodeContext *s = avctx->priv_data;
    int i, ret;

    if (avctx->channels != 1) {
        av_log(avctx, AV_LOG_ERROR, "Nellymoser supports only 1 channel\n");
        return AVERROR(EINVAL);
    }

    if (avctx->sample_rate != 8000 && avctx->sample_rate != 16000 &&
        avctx->sample_rate != 11025 &&
        avctx->sample_rate != 22050 && avctx->sample_rate != 44100 &&
        avctx->strict_std_compliance >= FF_COMPLIANCE_NORMAL) {
        av_log(avctx, AV_LOG_ERROR, "Nellymoser works only with 8000, 16000, 11025, 22050 and 44100 sample rate\n");
        return AVERROR(EINVAL);
    }

    avctx->frame_size = NELLY_SAMPLES;
    avctx->initial_padding = NELLY_BUF_LEN;
    ff_af_queue_init(avctx, &s->afq);
    s->avctx = avctx;
    if ((ret = ff_mdct_init(&s->mdct_ctx, 8, 0, 32768.0)) < 0)
        goto error;
