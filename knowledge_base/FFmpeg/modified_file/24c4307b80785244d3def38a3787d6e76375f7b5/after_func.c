void ff_vk_decode_free_params(void *opaque, uint8_t *data)
{
    FFVulkanDecodeShared *ctx = opaque;
    FFVulkanFunctions *vk = &ctx->s.vkfn;
    VkVideoSessionParametersKHR *par = (VkVideoSessionParametersKHR *)data;
    vk->DestroyVideoSessionParametersKHR(ctx->s.hwctx->act_dev, *par,
                                         ctx->s.hwctx->alloc);
    av_free(par);
}

int ff_vk_decode_uninit(AVCodecContext *avctx)
{
    FFVulkanDecodeContext *dec = avctx->internal->hwaccel_priv_data;
    av_buffer_pool_uninit(&dec->tmp_pool);
    av_buffer_unref(&dec->session_params);
    av_buffer_unref(&dec->shared_ref);
    return 0;
}

int ff_vk_decode_init(AVCodecContext *avctx)
{
    int err, qf, cxpos = 0, cypos = 0, nb_q = 0;
    VkResult ret;
    FFVulkanDecodeContext *dec = avctx->internal->hwaccel_priv_data;
    FFVulkanDecodeShared *ctx;
    FFVulkanDecodeProfileData *prof;
    FFVulkanContext *s;
    FFVulkanFunctions *vk;
    FFVkQueueFamilyCtx qf_dec;

    VkVideoDecodeH264SessionParametersCreateInfoKHR h264_params = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H264_SESSION_PARAMETERS_CREATE_INFO_KHR,
    };
    VkVideoDecodeH265SessionParametersCreateInfoKHR h265_params = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_H265_SESSION_PARAMETERS_CREATE_INFO_KHR,
    };
    VkVideoDecodeAV1SessionParametersCreateInfoMESA av1_params = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_DECODE_AV1_SESSION_PARAMETERS_CREATE_INFO_MESA,
    };
    VkVideoSessionParametersCreateInfoKHR session_params_create = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR,
        .pNext = avctx->codec_id == AV_CODEC_ID_H264 ? (void *)&h264_params :
                 avctx->codec_id == AV_CODEC_ID_HEVC ? (void *)&h265_params :
                 avctx->codec_id == AV_CODEC_ID_AV1  ? (void *)&av1_params  :
                 NULL,
    };
    VkVideoSessionCreateInfoKHR session_create = {
        .sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_CREATE_INFO_KHR,
    };
    VkSamplerYcbcrConversionCreateInfo yuv_sampler_info = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO,
        .components = ff_comp_identity_map,
        .ycbcrModel = VK_SAMPLER_YCBCR_MODEL_CONVERSION_RGB_IDENTITY,
        .ycbcrRange = avctx->color_range == AVCOL_RANGE_MPEG, /* Ignored */
    };

    err = ff_decode_get_hw_frames_ctx(avctx, AV_HWDEVICE_TYPE_VULKAN);
    if (err < 0)
        return err;

    /* Initialize contexts */
    ctx = (FFVulkanDecodeShared *)dec->shared_ref->data;
    prof = &ctx->profile_data;
    s = &ctx->s;
    vk = &ctx->s.vkfn;

    s->frames_ref = av_buffer_ref(avctx->hw_frames_ctx);
    s->frames = (AVHWFramesContext *)s->frames_ref->data;
    s->hwfc = s->frames->hwctx;

    s->device = (AVHWDeviceContext *)s->frames->device_ref->data;
    s->hwctx = s->device->hwctx;

    err = ff_vk_load_props(s);
    if (err < 0)
        goto fail;

    /* Create queue context */
    qf = ff_vk_qf_init(s, &qf_dec, VK_QUEUE_VIDEO_DECODE_BIT_KHR);

    /* Check for support */
    if (!(s->video_props[qf].videoCodecOperations &
          ff_vk_codec_map[avctx->codec_id].decode_op)) {
        av_log(avctx, AV_LOG_ERROR, "Decoding %s not supported on the given "
               "queue family %i!\n", avcodec_get_name(avctx->codec_id), qf);
        return AVERROR(EINVAL);
    }

    /* Enable queries if supported */
    if (s->query_props[qf].queryResultStatusSupport)
        nb_q = 1;

    session_create.flags = 0x0;
    session_create.queueFamilyIndex = s->hwctx->queue_family_decode_index;
    session_create.maxCodedExtent = prof->caps.maxCodedExtent;
    session_create.maxDpbSlots = prof->caps.maxDpbSlots;
    session_create.maxActiveReferencePictures = prof->caps.maxActiveReferencePictures;
    session_create.pictureFormat = s->hwfc->format[0];
    session_create.referencePictureFormat = session_create.pictureFormat;
    session_create.pStdHeaderVersion = dec_ext[avctx->codec_id];
    session_create.pVideoProfile = &prof->profile_list.pProfiles[0];

    /* Create decode exec context.
     * 2 async contexts per thread was experimentally determined to be optimal
     * for a majority of streams. */
    err = ff_vk_exec_pool_init(s, &qf_dec, &ctx->exec_pool, 2*avctx->thread_count,
                               nb_q, VK_QUERY_TYPE_RESULT_STATUS_ONLY_KHR, 0,
                               session_create.pVideoProfile);
    if (err < 0)
        goto fail;

    err = ff_vk_video_common_init(avctx, s, &ctx->common, &session_create);
    if (err < 0)
        goto fail;

    /* Get sampler */
    av_chroma_location_enum_to_pos(&cxpos, &cypos, avctx->chroma_sample_location);
    yuv_sampler_info.xChromaOffset = cxpos >> 7;
    yuv_sampler_info.yChromaOffset = cypos >> 7;
    yuv_sampler_info.format = s->hwfc->format[0];
    ret = vk->CreateSamplerYcbcrConversion(s->hwctx->act_dev, &yuv_sampler_info,
                                           s->hwctx->alloc, &ctx->yuv_sampler);
    if (ret != VK_SUCCESS) {
        err = AVERROR_EXTERNAL;
        goto fail;
    }

    /* If doing an out-of-place decoding, create a DPB pool */
    if (dec->dedicated_dpb || avctx->codec_id == AV_CODEC_ID_AV1) {
        AVHWFramesContext *dpb_frames;
        AVVulkanFramesContext *dpb_hwfc;

        ctx->dpb_hwfc_ref = av_hwframe_ctx_alloc(s->frames->device_ref);
        if (!ctx->dpb_hwfc_ref) {
            err = AVERROR(ENOMEM);
            goto fail;
        }

        dpb_frames = (AVHWFramesContext *)ctx->dpb_hwfc_ref->data;
        dpb_frames->format    = s->frames->format;
        dpb_frames->sw_format = s->frames->sw_format;
        dpb_frames->width     = s->frames->width;
        dpb_frames->height    = s->frames->height;

        dpb_hwfc = dpb_frames->hwctx;
        dpb_hwfc->create_pnext = (void *)&prof->profile_list;
        dpb_hwfc->format[0]    = s->hwfc->format[0];
        dpb_hwfc->tiling       = VK_IMAGE_TILING_OPTIMAL;
        dpb_hwfc->usage        = VK_IMAGE_USAGE_VIDEO_DECODE_DPB_BIT_KHR |
                                 VK_IMAGE_USAGE_SAMPLED_BIT; /* Shuts validator up. */

        if (dec->layered_dpb)
            dpb_hwfc->nb_layers = prof->caps.maxDpbSlots;

        err = av_hwframe_ctx_init(ctx->dpb_hwfc_ref);
        if (err < 0)
            goto fail;

        if (dec->layered_dpb) {
            ctx->layered_frame = vk_get_dpb_pool(ctx);
            if (!ctx->layered_frame) {
                err = AVERROR(ENOMEM);
                goto fail;
            }

            err = vk_decode_create_view(ctx, &ctx->layered_view, &ctx->layered_aspect,
                                        (AVVkFrame *)ctx->layered_frame->data[0],
                                        s->hwfc->format[0]);
            if (err < 0)
                goto fail;
        }
    }
