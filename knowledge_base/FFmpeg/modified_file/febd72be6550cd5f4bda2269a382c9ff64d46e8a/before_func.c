    ctx->data_type   = IEC61937_TRUEHD;
    ctx->pkt_offset  = 61440;
    ctx->out_buf     = ctx->hd_buf;
    ctx->out_bytes   = MAT_FRAME_SIZE;
    ctx->length_code = MAT_FRAME_SIZE;
    return 0;
}

static int spdif_write_header(AVFormatContext *s)
{
    IEC61937Context *ctx = s->priv_data;

    switch (s->streams[0]->codec->codec_id) {
    case CODEC_ID_AC3:
        ctx->header_info = spdif_header_ac3;
        break;
    case CODEC_ID_EAC3:
        ctx->header_info = spdif_header_eac3;
        break;
    case CODEC_ID_MP1:
    case CODEC_ID_MP2:
    case CODEC_ID_MP3:
        ctx->header_info = spdif_header_mpeg;
        break;
    case CODEC_ID_DTS:
        ctx->header_info = spdif_header_dts;
        break;
    case CODEC_ID_AAC:
        ctx->header_info = spdif_header_aac;
        break;
    case CODEC_ID_TRUEHD:
        ctx->header_info = spdif_header_truehd;
        ctx->hd_buf = av_malloc(MAT_FRAME_SIZE);
        if (!ctx->hd_buf)
            return AVERROR(ENOMEM);
        break;
    default:
        av_log(s, AV_LOG_ERROR, "codec not supported\n");
        return AVERROR_PATCHWELCOME;
    }
    return 0;
}

static int spdif_write_trailer(AVFormatContext *s)
{
    IEC61937Context *ctx = s->priv_data;
    av_freep(&ctx->buffer);
    av_freep(&ctx->hd_buf);
    return 0;
}

static int spdif_write_packet(struct AVFormatContext *s, AVPacket *pkt)
