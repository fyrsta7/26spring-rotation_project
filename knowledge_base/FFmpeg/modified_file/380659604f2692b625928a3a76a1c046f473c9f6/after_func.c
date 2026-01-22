    /* subtract offset from previous samples to use in prediction */
    if (command == FN_QLPC && coffset)
        for (i = -pred_order; i < 0; i++)
            s->decoded[channel][i] -= coffset;

    /* decode residual and do LPC prediction */
    init_sum = pred_order ? (command == FN_QLPC ? s->lpcqoffset : 0) : coffset;
    for (i = 0; i < s->blocksize; i++) {
        sum = init_sum;
        for (j = 0; j < pred_order; j++)
            sum += coeffs[j] * s->decoded[channel][i - j - 1];
        s->decoded[channel][i] = get_sr_golomb_shorten(&s->gb, residual_size) +
                                 (sum >> qshift);
    }

    /* add offset to current samples */
    if (command == FN_QLPC && coffset)
        for (i = 0; i < s->blocksize; i++)
            s->decoded[channel][i] += coffset;

    return 0;
}

static int read_header(ShortenContext *s)
{
    int i, ret;
    int maxnlpc = 0;
    /* shorten signature */
    if (get_bits_long(&s->gb, 32) != AV_RB32("ajkg")) {
        av_log(s->avctx, AV_LOG_ERROR, "missing shorten magic 'ajkg'\n");
        return AVERROR_INVALIDDATA;
    }

    s->lpcqoffset     = 0;
    s->blocksize      = DEFAULT_BLOCK_SIZE;
    s->nmean          = -1;
    s->version        = get_bits(&s->gb, 8);
    s->internal_ftype = get_uint(s, TYPESIZE);

    s->channels = get_uint(s, CHANSIZE);
    if (!s->channels) {
        av_log(s->avctx, AV_LOG_ERROR, "No channels reported\n");
        return AVERROR_INVALIDDATA;
    }
    if (s->channels > MAX_CHANNELS) {
        av_log(s->avctx, AV_LOG_ERROR, "too many channels: %d\n", s->channels);
        s->channels = 0;
        return AVERROR_INVALIDDATA;
    }
    s->avctx->channels = s->channels;

    /* get blocksize if version > 0 */
    if (s->version > 0) {
        int skip_bytes;
        unsigned blocksize;

        blocksize = get_uint(s, av_log2(DEFAULT_BLOCK_SIZE));
        if (!blocksize || blocksize > MAX_BLOCKSIZE) {
            av_log(s->avctx, AV_LOG_ERROR,
                   "invalid or unsupported block size: %d\n",
                   blocksize);
            return AVERROR(EINVAL);
        }
        s->blocksize = blocksize;

        maxnlpc  = get_uint(s, LPCQSIZE);
        if (maxnlpc > 1024U) {
            av_log(s->avctx, AV_LOG_ERROR, "maxnlpc is: %d\n", maxnlpc);
            return AVERROR_INVALIDDATA;
        }
        s->nmean = get_uint(s, 0);

        skip_bytes = get_uint(s, NSKIPSIZE);
        if ((unsigned)skip_bytes > get_bits_left(&s->gb)/8) {
            av_log(s->avctx, AV_LOG_ERROR, "invalid skip_bytes: %d\n", skip_bytes);
            return AVERROR_INVALIDDATA;
        }

        for (i = 0; i < skip_bytes; i++)
            skip_bits(&s->gb, 8);
    }
    s->nwrap = FFMAX(NWRAP, maxnlpc);

    if (s->version > 1)
        s->lpcqoffset = V2LPCQOFFSET;

    if (s->avctx->extradata_size > 0)
        goto end;

    if (get_ur_golomb_shorten(&s->gb, FNSIZE) != FN_VERBATIM) {
        av_log(s->avctx, AV_LOG_ERROR,
               "missing verbatim section at beginning of stream\n");
        return AVERROR_INVALIDDATA;
    }

    s->header_size = get_ur_golomb_shorten(&s->gb, VERBATIM_CKSIZE_SIZE);
    if (s->header_size >= OUT_BUFFER_SIZE ||
        s->header_size < CANONICAL_HEADER_SIZE) {
        av_log(s->avctx, AV_LOG_ERROR, "header is wrong size: %d\n",
               s->header_size);
        return AVERROR_INVALIDDATA;
    }

    for (i = 0; i < s->header_size; i++)
        s->header[i] = (char)get_ur_golomb_shorten(&s->gb, VERBATIM_BYTE_SIZE);

    if (AV_RL32(s->header) == MKTAG('R','I','F','F')) {
        if ((ret = decode_wave_header(s->avctx, s->header, s->header_size)) < 0)
            return ret;
    } else if (AV_RL32(s->header) == MKTAG('F','O','R','M')) {
