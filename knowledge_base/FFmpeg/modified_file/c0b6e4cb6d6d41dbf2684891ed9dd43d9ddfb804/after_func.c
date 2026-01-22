        goto exit;

    ret = CHECK_CU(cu->cuLaunchKernel(func,
                                      DIV_UP(dst_width, BLOCKX), DIV_UP(dst_height, BLOCKY), 1,
                                      BLOCKX, BLOCKY, 1,
                                      0, s->stream, args, NULL));

exit:
    if (tex_prev)
        CHECK_CU(cu->cuTexObjectDestroy(tex_prev));
    if (tex_cur)
        CHECK_CU(cu->cuTexObjectDestroy(tex_cur));
    if (tex_next)
        CHECK_CU(cu->cuTexObjectDestroy(tex_next));

    return ret;
}

static void filter(AVFilterContext *ctx, AVFrame *dst,
                   int parity, int tff)
{
    DeintCUDAContext *s = ctx->priv;
    YADIFContext *y = &s->yadif;
    CudaFunctions *cu = s->hwctx->internal->cuda_dl;
    CUcontext dummy;
    int i, ret;

    ret = CHECK_CU(cu->cuCtxPushCurrent(s->cu_ctx));
    if (ret < 0)
        return;

    for (i = 0; i < y->csp->nb_components; i++) {
        CUfunction func;
        CUarray_format format;
        int pixel_size, channels;
        const AVComponentDescriptor *comp = &y->csp->comp[i];

        if (comp->plane < i) {
            // We process planes as a whole, so don't reprocess
            // them for additional components
            continue;
        }

        pixel_size = (comp->depth + comp->shift) / 8;
        channels = comp->step / pixel_size;
        if (pixel_size > 2 || channels > 2) {
            av_log(ctx, AV_LOG_ERROR, "Unsupported pixel format: %s\n", y->csp->name);
            goto exit;
        }
        switch (pixel_size) {
        case 1:
            func = channels == 1 ? s->cu_func_uchar : s->cu_func_uchar2;
            format = CU_AD_FORMAT_UNSIGNED_INT8;
            break;
        case 2:
            func = channels == 1 ? s->cu_func_ushort : s->cu_func_ushort2;
            format = CU_AD_FORMAT_UNSIGNED_INT16;
            break;
        default:
            av_log(ctx, AV_LOG_ERROR, "Unsupported pixel format: %s\n", y->csp->name);
            goto exit;
        }
        av_log(ctx, AV_LOG_TRACE,
               "Deinterlacing plane %d: pixel_size: %d channels: %d\n",
               comp->plane, pixel_size, channels);
        call_kernel(ctx, func,
