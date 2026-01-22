     * obtained from the creator of the format.
     * Libavcodec has been assigned with the ID 0xF0.
     */
    AV_WB32(avctx->extradata, MKTAG(1, 0, 0, 0xF0));

    /*
     * Set the "original format"
     * Not used for anything during decoding.
     */
    AV_WL32(avctx->extradata + 4, original_format);

    /* Write 4 as the 'frame info size' */
    AV_WL32(avctx->extradata + 8, c->frame_info_size);

    /*
     * Set how many slices are going to be used.
     * Set one slice for now.
     */
