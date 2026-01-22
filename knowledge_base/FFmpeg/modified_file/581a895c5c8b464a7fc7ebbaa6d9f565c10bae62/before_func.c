                bd.ltp_gain    = ctx->ltp_gain[c];
                bd.lpc_cof     = ctx->lpc_cof[c];
                bd.quant_cof   = ctx->quant_cof[c];
                bd.raw_samples = ctx->raw_samples[c] + offset;
                bd.raw_other   = NULL;

                if ((ret = read_block(ctx, &bd)) < 0)
                    return ret;
                if ((ret = read_channel_data(ctx, ctx->chan_data[c], c)) < 0)
                    return ret;
            }

            for (c = 0; c < avctx->channels; c++) {
                ret = revert_channel_correlation(ctx, &bd, ctx->chan_data,
                                                 reverted_channels, offset, c);
                if (ret < 0)
                    return ret;
            }
            for (c = 0; c < avctx->channels; c++) {
                bd.const_block = ctx->const_block + c;
                bd.shift_lsbs  = ctx->shift_lsbs + c;
                bd.opt_order   = ctx->opt_order + c;
                bd.store_prev_samples = ctx->store_prev_samples + c;
                bd.use_ltp     = ctx->use_ltp + c;
                bd.ltp_lag     = ctx->ltp_lag + c;
                bd.ltp_gain    = ctx->ltp_gain[c];
                bd.lpc_cof     = ctx->lpc_cof[c];
                bd.quant_cof   = ctx->quant_cof[c];
                bd.raw_samples = ctx->raw_samples[c] + offset;

                if ((ret = decode_block(ctx, &bd)) < 0)
                    return ret;
            }

            memset(reverted_channels, 0, avctx->channels * sizeof(*reverted_channels));
            offset      += div_blocks[b];
            bd.ra_block  = 0;
        }

        // store carryover raw samples
        for (c = 0; c < avctx->channels; c++)
            memmove(ctx->raw_samples[c] - sconf->max_order,
                    ctx->raw_samples[c] - sconf->max_order + sconf->frame_length,
                    sizeof(*ctx->raw_samples[c]) * sconf->max_order);
    }

    if (sconf->floating) {
        read_diff_float_data(ctx, ra_frame);
    }

    if (get_bits_left(gb) < 0) {
        av_log(ctx->avctx, AV_LOG_ERROR, "Overread %d\n", -get_bits_left(gb));
        return AVERROR_INVALIDDATA;
    }

    return 0;
}


/** Decode an ALS frame.
 */
static int decode_frame(AVCodecContext *avctx, void *data, int *got_frame_ptr,
                        AVPacket *avpkt)
{
    ALSDecContext *ctx       = avctx->priv_data;
    AVFrame *frame           = data;
    ALSSpecificConfig *sconf = &ctx->sconf;
    const uint8_t *buffer    = avpkt->data;
    int buffer_size          = avpkt->size;
    int invalid_frame, ret;
    unsigned int c, sample, ra_frame, bytes_read, shift;

    if ((ret = init_get_bits8(&ctx->gb, buffer, buffer_size)) < 0)
        return ret;

    // In the case that the distance between random access frames is set to zero
    // (sconf->ra_distance == 0) no frame is treated as a random access frame.
    // For the first frame, if prediction is used, all samples used from the
    // previous frame are assumed to be zero.
    ra_frame = sconf->ra_distance && !(ctx->frame_id % sconf->ra_distance);

    // the last frame to decode might have a different length
    if (sconf->samples != 0xFFFFFFFF)
        ctx->cur_frame_length = FFMIN(sconf->samples - ctx->frame_id * (uint64_t) sconf->frame_length,
                                      sconf->frame_length);
    else
        ctx->cur_frame_length = sconf->frame_length;

    // decode the frame data
    if ((invalid_frame = read_frame_data(ctx, ra_frame)) < 0)
        av_log(ctx->avctx, AV_LOG_WARNING,
               "Reading frame data failed. Skipping RA unit.\n");

    ctx->frame_id++;

    /* get output buffer */
    frame->nb_samples = ctx->cur_frame_length;
    if ((ret = ff_get_buffer(avctx, frame, 0)) < 0)
        return ret;

    // transform decoded frame into output format
    #define INTERLEAVE_OUTPUT(bps)                                                   \
    {                                                                                \
        int##bps##_t *dest = (int##bps##_t*)frame->data[0];                          \
        shift = bps - ctx->avctx->bits_per_raw_sample;                               \
        if (!ctx->cs_switch) {                                                       \
            for (sample = 0; sample < ctx->cur_frame_length; sample++)               \
                for (c = 0; c < avctx->channels; c++)                                \
                    *dest++ = ctx->raw_samples[c][sample] * (1U << shift);            \
        } else {                                                                     \
            for (sample = 0; sample < ctx->cur_frame_length; sample++)               \
                for (c = 0; c < avctx->channels; c++)                                \
                    *dest++ = ctx->raw_samples[sconf->chan_pos[c]][sample] * (1U << shift); \
        }                                                                            \
    }

    if (ctx->avctx->bits_per_raw_sample <= 16) {
        INTERLEAVE_OUTPUT(16)
    } else {
        INTERLEAVE_OUTPUT(32)
    }

    // update CRC
    if (sconf->crc_enabled && (avctx->err_recognition & (AV_EF_CRCCHECK|AV_EF_CAREFUL))) {
        int swap = HAVE_BIGENDIAN != sconf->msb_first;
