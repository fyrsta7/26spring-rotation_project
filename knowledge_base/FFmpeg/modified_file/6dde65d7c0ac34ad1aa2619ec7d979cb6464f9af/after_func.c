        if (bit_alloc_stages[ch] > 1) {
            /* Compute excitation function, Compute masking curve, and
               Apply delta bit allocation */
            if (ff_ac3_bit_alloc_calc_mask(&s->bit_alloc_params, s->band_psd[ch],
                                           s->start_freq[ch],  s->end_freq[ch],
                                           s->fast_gain[ch],   (ch == s->lfe_ch),
                                           s->dba_mode[ch],    s->dba_nsegs[ch],
                                           s->dba_offsets[ch], s->dba_lengths[ch],
                                           s->dba_values[ch],  s->mask[ch])) {
                av_log(s->avctx, AV_LOG_ERROR, "error in bit allocation\n");
                return AVERROR_INVALIDDATA;
            }
        }
        if (bit_alloc_stages[ch] > 0) {
            /* Compute bit allocation */
            const uint8_t *bap_tab = s->channel_uses_aht[ch] ?
                                     ff_eac3_hebap_tab : ff_ac3_bap_tab;
            s->ac3dsp.bit_alloc_calc_bap(s->mask[ch], s->psd[ch],
                                      s->start_freq[ch], s->end_freq[ch],
                                      s->snr_offset[ch],
                                      s->bit_alloc_params.floor,
                                      bap_tab, s->bap[ch]);
        }
    }

    /* unused dummy data */
    if (s->skip_syntax && get_bits1(gbc)) {
        int skipl = get_bits(gbc, 9);
        skip_bits_long(gbc, 8 * skipl);
    }

    /* unpack the transform coefficients
       this also uncouples channels if coupling is in use. */
    decode_transform_coeffs(s, blk);

    /* TODO: generate enhanced coupling coordinates and uncouple */

    /* recover coefficients if rematrixing is in use */
    if (s->channel_mode == AC3_CHMODE_STEREO)
        do_rematrixing(s);

    /* apply scaling to coefficients (headroom, dynrng) */
    for (ch = 1; ch <= s->channels; ch++) {
        int audio_channel = 0;
        INTFLOAT gain;
        if (s->channel_mode == AC3_CHMODE_DUALMONO && ch <= 2)
            audio_channel = 2-ch;
        if (s->heavy_compression && s->compression_exists[audio_channel])
            gain = s->heavy_dynamic_range[audio_channel];
        else
            gain = s->dynamic_range[audio_channel];

#if USE_FIXED
        scale_coefs(s->transform_coeffs[ch], s->fixed_coeffs[ch], gain, 256);
#else
        if (s->target_level != 0)
          gain = gain * s->level_gain[audio_channel];
        gain *= 1.0 / 4194304.0f;
        s->fmt_conv.int32_to_float_fmul_scalar(s->transform_coeffs[ch],
                                               s->fixed_coeffs[ch], gain, 256);
#endif
    }

    /* apply spectral extension to high frequency bins */
    if (CONFIG_EAC3_DECODER && s->spx_in_use) {
        ff_eac3_apply_spectral_extension(s);
    }

    /* downmix and MDCT. order depends on whether block switching is used for
       any channel in this block. this is because coefficients for the long
       and short transforms cannot be mixed. */
    downmix_output = s->channels != s->out_channels &&
                     !((s->output_mode & AC3_OUTPUT_LFEON) &&
                     s->fbw_channels == s->out_channels);
    if (different_transforms) {
        /* the delay samples have already been downmixed, so we upmix the delay
           samples in order to reconstruct all channels before downmixing. */
        if (s->downmixed) {
            s->downmixed = 0;
            ac3_upmix_delay(s);
        }

        do_imdct(s, s->channels, offset);

        if (downmix_output) {
#if USE_FIXED
            ac3_downmix_c_fixed16(s->outptr, s->downmix_coeffs,
                              s->out_channels, s->fbw_channels, 256);
#else
            ff_ac3dsp_downmix(&s->ac3dsp, s->outptr, s->downmix_coeffs,
                              s->out_channels, s->fbw_channels, 256);
#endif
        }
    } else {
        if (downmix_output) {
            AC3_RENAME(ff_ac3dsp_downmix)(&s->ac3dsp, s->xcfptr + 1, s->downmix_coeffs,
                                          s->out_channels, s->fbw_channels, 256);
        }

        if (downmix_output && !s->downmixed) {
            s->downmixed = 1;
            AC3_RENAME(ff_ac3dsp_downmix)(&s->ac3dsp, s->dlyptr, s->downmix_coeffs,
                                          s->out_channels, s->fbw_channels, 128);
        }

        do_imdct(s, s->out_channels, offset);
    }

    return 0;
}

/**
 * Decode a single AC-3 frame.
 */
static int ac3_decode_frame(AVCodecContext * avctx, void *data,
                            int *got_frame_ptr, AVPacket *avpkt)
{
    AVFrame *frame     = data;
    const uint8_t *buf = avpkt->data;
    int buf_size, full_buf_size = avpkt->size;
    AC3DecodeContext *s = avctx->priv_data;
    int blk, ch, err, offset, ret;
    int i;
    int skip = 0, got_independent_frame = 0;
    const uint8_t *channel_map;
    uint8_t extended_channel_map[EAC3_MAX_CHANNELS];
    const SHORTFLOAT *output[AC3_MAX_CHANNELS];
    enum AVMatrixEncoding matrix_encoding;
    AVDownmixInfo *downmix_info;

    s->superframe_size = 0;

    buf_size = full_buf_size;
    for (i = 1; i < buf_size; i += 2) {
        if (buf[i] == 0x77 || buf[i] == 0x0B) {
            if ((buf[i] ^ buf[i-1]) == (0x77 ^ 0x0B)) {
                i--;
                break;
            } else if ((buf[i] ^ buf[i+1]) == (0x77 ^ 0x0B)) {
                break;
            }
        }
    }
    if (i >= buf_size)
        return AVERROR_INVALIDDATA;
    buf += i;
    buf_size -= i;

    /* copy input buffer to decoder context to avoid reading past the end
       of the buffer, which can be caused by a damaged input stream. */
    if (buf_size >= 2 && AV_RB16(buf) == 0x770B) {
        // seems to be byte-swapped AC-3
        int cnt = FFMIN(buf_size, AC3_FRAME_BUFFER_SIZE) >> 1;
        s->bdsp.bswap16_buf((uint16_t *) s->input_buffer,
                            (const uint16_t *) buf, cnt);
    } else
        memcpy(s->input_buffer, buf, FFMIN(buf_size, AC3_FRAME_BUFFER_SIZE));

    /* if consistent noise generation is enabled, seed the linear feedback generator
     * with the contents of the AC-3 frame so that the noise is identical across
     * decodes given the same AC-3 frame data, for use with non-linear edititing software. */
    if (s->consistent_noise_generation)
        av_lfg_init_from_data(&s->dith_state, s->input_buffer, FFMIN(buf_size, AC3_FRAME_BUFFER_SIZE));

    buf = s->input_buffer;
dependent_frame:
    /* initialize the GetBitContext with the start of valid AC-3 Frame */
    if ((ret = init_get_bits8(&s->gbc, buf, buf_size)) < 0)
        return ret;

    /* parse the syncinfo */
    err = parse_frame_header(s);

    if (err) {
        switch (err) {
        case AAC_AC3_PARSE_ERROR_SYNC:
            av_log(avctx, AV_LOG_ERROR, "frame sync error\n");
            return AVERROR_INVALIDDATA;
        case AAC_AC3_PARSE_ERROR_BSID:
            av_log(avctx, AV_LOG_ERROR, "invalid bitstream id\n");
            break;
        case AAC_AC3_PARSE_ERROR_SAMPLE_RATE:
            av_log(avctx, AV_LOG_ERROR, "invalid sample rate\n");
            break;
        case AAC_AC3_PARSE_ERROR_FRAME_SIZE:
            av_log(avctx, AV_LOG_ERROR, "invalid frame size\n");
            break;
        case AAC_AC3_PARSE_ERROR_FRAME_TYPE:
            /* skip frame if CRC is ok. otherwise use error concealment. */
            /* TODO: add support for substreams */
            if (s->substreamid) {
                av_log(avctx, AV_LOG_DEBUG,
                       "unsupported substream %d: skipping frame\n",
                       s->substreamid);
                *got_frame_ptr = 0;
                return buf_size;
            } else {
                av_log(avctx, AV_LOG_ERROR, "invalid frame type\n");
            }
            break;
        case AAC_AC3_PARSE_ERROR_CRC:
        case AAC_AC3_PARSE_ERROR_CHANNEL_CFG:
            break;
        default: // Normal AVERROR do not try to recover.
            *got_frame_ptr = 0;
            return err;
        }
    } else {
        /* check that reported frame size fits in input buffer */
        if (s->frame_size > buf_size) {
            av_log(avctx, AV_LOG_ERROR, "incomplete frame\n");
            err = AAC_AC3_PARSE_ERROR_FRAME_SIZE;
        } else if (avctx->err_recognition & (AV_EF_CRCCHECK|AV_EF_CAREFUL)) {
            /* check for crc mismatch */
            if (av_crc(av_crc_get_table(AV_CRC_16_ANSI), 0, &buf[2],
                       s->frame_size - 2)) {
                av_log(avctx, AV_LOG_ERROR, "frame CRC mismatch\n");
                if (avctx->err_recognition & AV_EF_EXPLODE)
                    return AVERROR_INVALIDDATA;
                err = AAC_AC3_PARSE_ERROR_CRC;
            }
        }
    }

    if (s->frame_type == EAC3_FRAME_TYPE_DEPENDENT && !got_independent_frame) {
        av_log(avctx, AV_LOG_WARNING, "Ignoring dependent frame without independent frame.\n");
        *got_frame_ptr = 0;
        return FFMIN(full_buf_size, s->frame_size);
    }

    /* channel config */
    if (!err || (s->channels && s->out_channels != s->channels)) {
        s->out_channels = s->channels;
        s->output_mode  = s->channel_mode;
        if (s->lfe_on)
            s->output_mode |= AC3_OUTPUT_LFEON;
        if (s->channels > 1 &&
            avctx->request_channel_layout == AV_CH_LAYOUT_MONO) {
            s->out_channels = 1;
            s->output_mode  = AC3_CHMODE_MONO;
        } else if (s->channels > 2 &&
                   avctx->request_channel_layout == AV_CH_LAYOUT_STEREO) {
            s->out_channels = 2;
            s->output_mode  = AC3_CHMODE_STEREO;
        }

        s->loro_center_mix_level   = gain_levels[s->  center_mix_level];
        s->loro_surround_mix_level = gain_levels[s->surround_mix_level];
        s->ltrt_center_mix_level   = LEVEL_MINUS_3DB;
        s->ltrt_surround_mix_level = LEVEL_MINUS_3DB;
        /* set downmixing coefficients if needed */
        if (s->channels != s->out_channels && !((s->output_mode & AC3_OUTPUT_LFEON) &&
                s->fbw_channels == s->out_channels)) {
            if ((ret = set_downmix_coeffs(s)) < 0) {
                av_log(avctx, AV_LOG_ERROR, "error setting downmix coeffs\n");
                return ret;
            }
        }
    } else if (!s->channels) {
        av_log(avctx, AV_LOG_ERROR, "unable to determine channel mode\n");
        return AVERROR_INVALIDDATA;
    }
    avctx->channels = s->out_channels;
    avctx->channel_layout = avpriv_ac3_channel_layout_tab[s->output_mode & ~AC3_OUTPUT_LFEON];
    if (s->output_mode & AC3_OUTPUT_LFEON)
        avctx->channel_layout |= AV_CH_LOW_FREQUENCY;

    /* set audio service type based on bitstream mode for AC-3 */
    avctx->audio_service_type = s->bitstream_mode;
    if (s->bitstream_mode == 0x7 && s->channels > 1)
        avctx->audio_service_type = AV_AUDIO_SERVICE_TYPE_KARAOKE;

    /* decode the audio blocks */
    channel_map = ff_ac3_dec_channel_map[s->output_mode & ~AC3_OUTPUT_LFEON][s->lfe_on];
    offset = s->frame_type == EAC3_FRAME_TYPE_DEPENDENT ? AC3_MAX_CHANNELS : 0;
    for (ch = 0; ch < AC3_MAX_CHANNELS; ch++) {
        output[ch] = s->output[ch + offset];
        s->outptr[ch] = s->output[ch + offset];
    }
    for (ch = 0; ch < s->channels; ch++) {
        if (ch < s->out_channels)
            s->outptr[channel_map[ch]] = s->output_buffer[ch + offset];
    }
    for (blk = 0; blk < s->num_blocks; blk++) {
        if (!err && decode_audio_block(s, blk, offset)) {
            av_log(avctx, AV_LOG_ERROR, "error decoding the audio block\n");
            err = 1;
        }
        if (err)
            for (ch = 0; ch < s->out_channels; ch++)
                memcpy(s->output_buffer[ch + offset] + AC3_BLOCK_SIZE*blk, output[ch], AC3_BLOCK_SIZE*sizeof(SHORTFLOAT));
        for (ch = 0; ch < s->out_channels; ch++)
            output[ch] = s->outptr[channel_map[ch]];
        for (ch = 0; ch < s->out_channels; ch++) {
            if (!ch || channel_map[ch])
                s->outptr[channel_map[ch]] += AC3_BLOCK_SIZE;
        }
    }

    /* keep last block for error concealment in next frame */
    for (ch = 0; ch < s->out_channels; ch++)
        memcpy(s->output[ch + offset], output[ch], AC3_BLOCK_SIZE*sizeof(SHORTFLOAT));

    /* check if there is dependent frame */
    if (buf_size > s->frame_size) {
        AC3HeaderInfo hdr;
        int err;

        if (buf_size - s->frame_size <= 16) {
            skip = buf_size - s->frame_size;
            goto skip;
        }

        if ((ret = init_get_bits8(&s->gbc, buf + s->frame_size, buf_size - s->frame_size)) < 0)
            return ret;

        err = ff_ac3_parse_header(&s->gbc, &hdr);
        if (err)
            return err;

        if (hdr.frame_type == EAC3_FRAME_TYPE_DEPENDENT) {
            if (hdr.num_blocks != s->num_blocks || s->sample_rate != hdr.sample_rate) {
                av_log(avctx, AV_LOG_WARNING, "Ignoring non-compatible dependent frame.\n");
            } else {
                buf += s->frame_size;
                buf_size -= s->frame_size;
                s->prev_output_mode = s->output_mode;
                s->prev_bit_rate = s->bit_rate;
                got_independent_frame = 1;
                goto dependent_frame;
            }
        }
    }
skip:

    frame->decode_error_flags = err ? FF_DECODE_ERROR_INVALID_BITSTREAM : 0;

    /* if frame is ok, set audio parameters */
    if (!err) {
        avctx->sample_rate = s->sample_rate;
        avctx->bit_rate    = s->bit_rate + s->prev_bit_rate;
    }

    for (ch = 0; ch < EAC3_MAX_CHANNELS; ch++)
        extended_channel_map[ch] = ch;

    if (s->frame_type == EAC3_FRAME_TYPE_DEPENDENT) {
        uint64_t ich_layout = avpriv_ac3_channel_layout_tab[s->prev_output_mode & ~AC3_OUTPUT_LFEON];
        int channel_map_size = ff_ac3_channels_tab[s->output_mode & ~AC3_OUTPUT_LFEON] + s->lfe_on;
        uint64_t channel_layout;
        int extend = 0;

        if (s->prev_output_mode & AC3_OUTPUT_LFEON)
            ich_layout |= AV_CH_LOW_FREQUENCY;

