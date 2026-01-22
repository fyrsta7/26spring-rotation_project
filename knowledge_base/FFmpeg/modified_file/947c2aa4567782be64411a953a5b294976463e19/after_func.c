
    qf->surface_internal.Data.PitchLow  = qf->frame->linesize[0];
    qf->surface_internal.Data.Y         = qf->frame->data[0];
    qf->surface_internal.Data.UV        = qf->frame->data[1];
    qf->surface_internal.Data.TimeStamp = av_rescale_q(frame->pts, q->avctx->time_base, (AVRational){1, 90000});

    qf->surface = &qf->surface_internal;

    *surface = qf->surface;

    return 0;
}

static void print_interlace_msg(AVCodecContext *avctx, QSVEncContext *q)
{
    if (q->param.mfx.CodecId == MFX_CODEC_AVC) {
        if (q->param.mfx.CodecProfile == MFX_PROFILE_AVC_BASELINE ||
            q->param.mfx.CodecLevel < MFX_LEVEL_AVC_21 ||
            q->param.mfx.CodecLevel > MFX_LEVEL_AVC_41)
            av_log(avctx, AV_LOG_WARNING,
                   "Interlaced coding is supported"
                   " at Main/High Profile Level 2.1-4.1\n");
    }
}

int ff_qsv_encode(AVCodecContext *avctx, QSVEncContext *q,
                  AVPacket *pkt, const AVFrame *frame, int *got_packet)
{
    AVPacket new_pkt = { 0 };
    mfxBitstream *bs;

    mfxFrameSurface1 *surf = NULL;
    mfxSyncPoint sync      = NULL;
    int ret;

    if (frame) {
        ret = submit_frame(q, frame, &surf);
        if (ret < 0) {
            av_log(avctx, AV_LOG_ERROR, "Error submitting the frame for encoding.\n");
            return ret;
        }
    }

    ret = av_new_packet(&new_pkt, q->packet_size);
    if (ret < 0) {
        av_log(avctx, AV_LOG_ERROR, "Error allocating the output packet\n");
        return ret;
    }

    bs = av_mallocz(sizeof(*bs));
    if (!bs) {
        av_packet_unref(&new_pkt);
        return AVERROR(ENOMEM);
    }
    bs->Data      = new_pkt.data;
    bs->MaxLength = new_pkt.size;

    do {
        ret = MFXVideoENCODE_EncodeFrameAsync(q->session, NULL, surf, bs, &sync);
        if (ret == MFX_WRN_DEVICE_BUSY) {
            av_usleep(500);
            continue;
        }
        break;
    } while ( 1 );

    if (ret < 0) {
        av_packet_unref(&new_pkt);
        av_freep(&bs);
        if (ret == MFX_ERR_MORE_DATA)
            return 0;
        av_log(avctx, AV_LOG_ERROR, "EncodeFrameAsync returned %d\n", ret);
        return ff_qsv_error(ret);
    }

    if (ret == MFX_WRN_INCOMPATIBLE_VIDEO_PARAM) {
        if (frame->interlaced_frame)
            print_interlace_msg(avctx, q);
        else
            av_log(avctx, AV_LOG_WARNING,
                   "EncodeFrameAsync returned 'incompatible param' code\n");
    }
    if (sync) {
        av_fifo_generic_write(q->async_fifo, &new_pkt, sizeof(new_pkt), NULL);
        av_fifo_generic_write(q->async_fifo, &sync,    sizeof(sync),    NULL);
        av_fifo_generic_write(q->async_fifo, &bs,      sizeof(bs),    NULL);
    } else {
        av_packet_unref(&new_pkt);
        av_freep(&bs);
    }

    if (!av_fifo_space(q->async_fifo) ||
        (!frame && av_fifo_size(q->async_fifo))) {
        av_fifo_generic_read(q->async_fifo, &new_pkt, sizeof(new_pkt), NULL);
        av_fifo_generic_read(q->async_fifo, &sync,    sizeof(sync),    NULL);
        av_fifo_generic_read(q->async_fifo, &bs,      sizeof(bs),      NULL);

        MFXVideoCORE_SyncOperation(q->session, sync, 60000);

        new_pkt.dts  = av_rescale_q(bs->DecodeTimeStamp, (AVRational){1, 90000}, avctx->time_base);
        new_pkt.pts  = av_rescale_q(bs->TimeStamp,       (AVRational){1, 90000}, avctx->time_base);
        new_pkt.size = bs->DataLength;

        if (bs->FrameType & MFX_FRAMETYPE_IDR ||
            bs->FrameType & MFX_FRAMETYPE_xIDR)
            new_pkt.flags |= AV_PKT_FLAG_KEY;

#if FF_API_CODED_FRAME
FF_DISABLE_DEPRECATION_WARNINGS
        if (bs->FrameType & MFX_FRAMETYPE_I || bs->FrameType & MFX_FRAMETYPE_xI)
            avctx->coded_frame->pict_type = AV_PICTURE_TYPE_I;
        else if (bs->FrameType & MFX_FRAMETYPE_P || bs->FrameType & MFX_FRAMETYPE_xP)
            avctx->coded_frame->pict_type = AV_PICTURE_TYPE_P;
        else if (bs->FrameType & MFX_FRAMETYPE_B || bs->FrameType & MFX_FRAMETYPE_xB)
            avctx->coded_frame->pict_type = AV_PICTURE_TYPE_B;
FF_ENABLE_DEPRECATION_WARNINGS
#endif

