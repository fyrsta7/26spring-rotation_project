        ff_qsv_print_warning(avctx, ret,
                             "Warning in encoder initialization");

    switch (avctx->codec_id) {
    case AV_CODEC_ID_MJPEG:
        ret = qsv_retrieve_enc_jpeg_params(avctx, q);
        break;
    default:
        ret = qsv_retrieve_enc_params(avctx, q);
        break;
    }
    if (ret < 0) {
        av_log(avctx, AV_LOG_ERROR, "Error retrieving encoding parameters.\n");
