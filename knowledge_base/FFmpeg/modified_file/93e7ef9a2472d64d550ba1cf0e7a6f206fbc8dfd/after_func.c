    if (!avpriv_flac_is_extradata_valid(avctx, &format, &streaminfo))
        return -1;

    /* initialize based on the demuxer-supplied streamdata header */
    avpriv_flac_parse_streaminfo(avctx, (FLACStreaminfo *)s, streaminfo);
    if (s->bps > 16)
        avctx->sample_fmt = AV_SAMPLE_FMT_S32;
    else
        avctx->sample_fmt = AV_SAMPLE_FMT_S16;
    allocate_buffers(s);
