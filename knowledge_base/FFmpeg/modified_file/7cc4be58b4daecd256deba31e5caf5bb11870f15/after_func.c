

/**
 * Finalize encoding and free any memory allocated by the encoder.
 */
static av_cold int ac3_encode_close(AVCodecContext *avctx)
{
    int blk, ch;
    AC3EncodeContext *s = avctx->priv_data;

    for (ch = 0; ch < s->channels; ch++)
        av_freep(&s->planar_samples[ch]);
    av_freep(&s->planar_samples);
    av_freep(&s->bap_buffer);
    av_freep(&s->bap1_buffer);
    av_freep(&s->mdct_coef_buffer);
    av_freep(&s->fixed_coef_buffer);
    av_freep(&s->exp_buffer);
    av_freep(&s->grouped_exp_buffer);
    av_freep(&s->psd_buffer);
    av_freep(&s->band_psd_buffer);
    av_freep(&s->mask_buffer);
    av_freep(&s->qmant_buffer);
    for (blk = 0; blk < AC3_MAX_BLOCKS; blk++) {
        AC3Block *block = &s->blocks[blk];
        av_freep(&block->bap);
        av_freep(&block->mdct_coef);
        av_freep(&block->fixed_coef);
        av_freep(&block->exp);
        av_freep(&block->grouped_exp);
        av_freep(&block->psd);
        av_freep(&block->band_psd);
        av_freep(&block->mask);
        av_freep(&block->qmant);
    }

    mdct_end(&s->mdct);

    av_freep(&avctx->coded_frame);
    return 0;
}


/**
 * Set channel information during initialization.
 */
static av_cold int set_channel_info(AC3EncodeContext *s, int channels,
                                    int64_t *channel_layout)
{
    int ch_layout;

    if (channels < 1 || channels > AC3_MAX_CHANNELS)
        return AVERROR(EINVAL);
    if ((uint64_t)*channel_layout > 0x7FF)
        return AVERROR(EINVAL);
    ch_layout = *channel_layout;
    if (!ch_layout)
        ch_layout = avcodec_guess_channel_layout(channels, CODEC_ID_AC3, NULL);
    if (av_get_channel_layout_nb_channels(ch_layout) != channels)
        return AVERROR(EINVAL);

    s->lfe_on       = !!(ch_layout & AV_CH_LOW_FREQUENCY);
    s->channels     = channels;
    s->fbw_channels = channels - s->lfe_on;
    s->lfe_channel  = s->lfe_on ? s->fbw_channels : -1;
    if (s->lfe_on)
        ch_layout -= AV_CH_LOW_FREQUENCY;

    switch (ch_layout) {
    case AV_CH_LAYOUT_MONO:           s->channel_mode = AC3_CHMODE_MONO;   break;
    case AV_CH_LAYOUT_STEREO:         s->channel_mode = AC3_CHMODE_STEREO; break;
    case AV_CH_LAYOUT_SURROUND:       s->channel_mode = AC3_CHMODE_3F;     break;
    case AV_CH_LAYOUT_2_1:            s->channel_mode = AC3_CHMODE_2F1R;   break;
    case AV_CH_LAYOUT_4POINT0:        s->channel_mode = AC3_CHMODE_3F1R;   break;
    case AV_CH_LAYOUT_QUAD:
    case AV_CH_LAYOUT_2_2:            s->channel_mode = AC3_CHMODE_2F2R;   break;
    case AV_CH_LAYOUT_5POINT0:
    case AV_CH_LAYOUT_5POINT0_BACK:   s->channel_mode = AC3_CHMODE_3F2R;   break;
    default:
        return AVERROR(EINVAL);
    }

    s->channel_map  = ff_ac3_enc_channel_map[s->channel_mode][s->lfe_on];
    *channel_layout = ch_layout;
    if (s->lfe_on)
        *channel_layout |= AV_CH_LOW_FREQUENCY;

    return 0;
