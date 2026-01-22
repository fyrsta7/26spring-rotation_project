        avctx->pix_fmt = PIX_FMT_PAL8;
    } else if (avctx->bits_per_coded_sample <= 32) {
        avctx->pix_fmt = PIX_FMT_BGR32;
    } else {
        return AVERROR_INVALIDDATA;
    }

    if ((err = avcodec_check_dimensions(avctx, avctx->width, avctx->height)))
        return err;
