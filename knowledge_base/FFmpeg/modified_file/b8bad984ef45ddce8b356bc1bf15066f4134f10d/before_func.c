    dsp->get_pixels(ctx->blocks[2], ptr_u,      ctx->m.uvlinesize);
    dsp->get_pixels(ctx->blocks[3], ptr_v,      ctx->m.uvlinesize);

    if (mb_y+1 == ctx->m.mb_height && ctx->m.avctx->height == 1080) {
        if (ctx->interlaced) {
            ctx->get_pixels_8x4_sym(ctx->blocks[4], ptr_y + ctx->dct_y_offset,      ctx->m.linesize);
            ctx->get_pixels_8x4_sym(ctx->blocks[5], ptr_y + ctx->dct_y_offset + bw, ctx->m.linesize);
            ctx->get_pixels_8x4_sym(ctx->blocks[6], ptr_u + ctx->dct_uv_offset,     ctx->m.uvlinesize);
