    dsp->get_pixels(ctx->blocks[2], ptr_u,      ctx->m.uvlinesize);
    dsp->get_pixels(ctx->blocks[3], ptr_v,      ctx->m.uvlinesize);

    if (mb_y+1 == ctx->m.mb_height && ctx->m.avctx->height == 1080) {
        if (ctx->interlaced) {
