    if (ctx->internal_error) {
        av_log(avctx, AV_LOG_ERROR, "cuvid decode callback error\n");
        ret = ctx->internal_error;
        goto error;
    }

error:
    eret = CHECK_CU(ctx->cudl->cuCtxPopCurrent(&dummy));

    if (eret < 0)
        return eret;
    else if (ret < 0)
        return ret;
    else if (is_flush)
        return AVERROR_EOF;
    else
        return 0;
}

static int cuvid_output_frame(AVCodecContext *avctx, AVFrame *frame)
{
    CuvidContext *ctx = avctx->priv_data;
    AVHWDeviceContext *device_ctx = (AVHWDeviceContext*)ctx->hwdevice->data;
    AVCUDADeviceContext *device_hwctx = device_ctx->hwctx;
    CUcontext dummy, cuda_ctx = device_hwctx->cuda_ctx;
    CUdeviceptr mapped_frame = 0;
    int ret = 0, eret = 0;

    av_log(avctx, AV_LOG_TRACE, "cuvid_output_frame\n");

    if (ctx->decoder_flushing) {
        ret = cuvid_decode_packet(avctx, NULL);
        if (ret < 0 && ret != AVERROR_EOF)
            return ret;
    }

    if (!cuvid_is_buffer_full(avctx)) {
        AVPacket pkt = {0};
        ret = ff_decode_get_packet(avctx, &pkt);
        if (ret < 0 && ret != AVERROR_EOF)
            return ret;
        ret = cuvid_decode_packet(avctx, &pkt);
        av_packet_unref(&pkt);
        // cuvid_is_buffer_full() should avoid this.
        if (ret == AVERROR(EAGAIN))
            ret = AVERROR_EXTERNAL;
        if (ret < 0 && ret != AVERROR_EOF)
            return ret;
    }

    ret = CHECK_CU(ctx->cudl->cuCtxPushCurrent(cuda_ctx));
    if (ret < 0)
        return ret;

    if (av_fifo_size(ctx->frame_queue)) {
        const AVPixFmtDescriptor *pixdesc;
        CuvidParsedFrame parsed_frame;
        CUVIDPROCPARAMS params;
        unsigned int pitch = 0;
        int offset = 0;
        int i;

        av_fifo_generic_read(ctx->frame_queue, &parsed_frame, sizeof(CuvidParsedFrame), NULL);

        memset(&params, 0, sizeof(params));
        params.progressive_frame = parsed_frame.dispinfo.progressive_frame;
        params.second_field = parsed_frame.second_field;
        params.top_field_first = parsed_frame.dispinfo.top_field_first;

        ret = CHECK_CU(ctx->cvdl->cuvidMapVideoFrame(ctx->cudecoder, parsed_frame.dispinfo.picture_index, &mapped_frame, &pitch, &params));
        if (ret < 0)
            goto error;

        if (avctx->pix_fmt == AV_PIX_FMT_CUDA) {
            ret = av_hwframe_get_buffer(ctx->hwframe, frame, 0);
            if (ret < 0) {
                av_log(avctx, AV_LOG_ERROR, "av_hwframe_get_buffer failed\n");
                goto error;
            }

            ret = ff_decode_frame_props(avctx, frame);
            if (ret < 0) {
                av_log(avctx, AV_LOG_ERROR, "ff_decode_frame_props failed\n");
                goto error;
            }

            pixdesc = av_pix_fmt_desc_get(avctx->sw_pix_fmt);

            for (i = 0; i < pixdesc->nb_components; i++) {
                int height = avctx->height >> (i ? pixdesc->log2_chroma_h : 0);
                CUDA_MEMCPY2D cpy = {
                    .srcMemoryType = CU_MEMORYTYPE_DEVICE,
                    .dstMemoryType = CU_MEMORYTYPE_DEVICE,
                    .srcDevice     = mapped_frame,
                    .dstDevice     = (CUdeviceptr)frame->data[i],
                    .srcPitch      = pitch,
                    .dstPitch      = frame->linesize[i],
                    .srcY          = offset,
                    .WidthInBytes  = FFMIN(pitch, frame->linesize[i]),
                    .Height        = height,
                };

                ret = CHECK_CU(ctx->cudl->cuMemcpy2DAsync(&cpy, device_hwctx->stream));
                if (ret < 0)
                    goto error;

                offset += height;
            }

            ret = CHECK_CU(ctx->cudl->cuStreamSynchronize(device_hwctx->stream));
            if (ret < 0)
                goto error;
        } else if (avctx->pix_fmt == AV_PIX_FMT_NV12      ||
                   avctx->pix_fmt == AV_PIX_FMT_P010      ||
                   avctx->pix_fmt == AV_PIX_FMT_P016      ||
                   avctx->pix_fmt == AV_PIX_FMT_YUV444P   ||
                   avctx->pix_fmt == AV_PIX_FMT_YUV444P16) {
            unsigned int offset = 0;
            AVFrame *tmp_frame = av_frame_alloc();
            if (!tmp_frame) {
                av_log(avctx, AV_LOG_ERROR, "av_frame_alloc failed\n");
                ret = AVERROR(ENOMEM);
                goto error;
            }

            pixdesc = av_pix_fmt_desc_get(avctx->sw_pix_fmt);

            tmp_frame->format        = AV_PIX_FMT_CUDA;
            tmp_frame->hw_frames_ctx = av_buffer_ref(ctx->hwframe);
            tmp_frame->width         = avctx->width;
            tmp_frame->height        = avctx->height;

            /*
             * Note that the following logic would not work for three plane
             * YUV420 because the pitch value is different for the chroma
             * planes.
             */
            for (i = 0; i < pixdesc->nb_components; i++) {
                tmp_frame->data[i]     = (uint8_t*)mapped_frame + offset;
                tmp_frame->linesize[i] = pitch;
                offset += pitch * (avctx->height >> (i ? pixdesc->log2_chroma_h : 0));
            }

            ret = ff_get_buffer(avctx, frame, 0);
            if (ret < 0) {
                av_log(avctx, AV_LOG_ERROR, "ff_get_buffer failed\n");
                av_frame_free(&tmp_frame);
                goto error;
            }

            ret = av_hwframe_transfer_data(frame, tmp_frame, 0);
            if (ret) {
                av_log(avctx, AV_LOG_ERROR, "av_hwframe_transfer_data failed\n");
                av_frame_free(&tmp_frame);
                goto error;
            }
            av_frame_free(&tmp_frame);
        } else {
            ret = AVERROR_BUG;
            goto error;
        }

        frame->key_frame = ctx->key_frame[parsed_frame.dispinfo.picture_index];
        frame->width = avctx->width;
        frame->height = avctx->height;
        if (avctx->pkt_timebase.num && avctx->pkt_timebase.den)
            frame->pts = av_rescale_q(parsed_frame.dispinfo.timestamp, (AVRational){1, 10000000}, avctx->pkt_timebase);
        else
            frame->pts = parsed_frame.dispinfo.timestamp;

        if (parsed_frame.second_field) {
            if (ctx->prev_pts == INT64_MIN) {
                ctx->prev_pts = frame->pts;
                frame->pts += (avctx->pkt_timebase.den * avctx->framerate.den) / (avctx->pkt_timebase.num * avctx->framerate.num);
            } else {
                int pts_diff = (frame->pts - ctx->prev_pts) / 2;
                ctx->prev_pts = frame->pts;
                frame->pts += pts_diff;
            }
        }

        /* CUVIDs opaque reordering breaks the internal pkt logic.
         * So set pkt_pts and clear all the other pkt_ fields.
         */
#if FF_API_PKT_PTS
FF_DISABLE_DEPRECATION_WARNINGS
        frame->pkt_pts = frame->pts;
FF_ENABLE_DEPRECATION_WARNINGS
#endif
