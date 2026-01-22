    av_freep(&ctx->inflated_buf);

    return 0;
}

static av_cold int screenpresso_init(AVCodecContext *avctx)
{
    ScreenpressoContext *ctx = avctx->priv_data;

    /* These needs to be set to estimate uncompressed buffer */
    int ret = av_image_check_size(avctx->width, avctx->height, 0, avctx);
