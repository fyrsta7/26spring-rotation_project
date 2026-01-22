{
    decoder_sys_t *sys = dec->p_sys;

    vlc_fourcc_t fourcc = FindVlcChroma(frame->format);
    if (!fourcc)
    {
        const char *name = av_get_pix_fmt_name(frame->format);

        msg_Err(dec, "Unsupported decoded output format %d (%s)",
                sys->p_context->pix_fmt, (name != NULL) ? name : "unknown");
        return VLC_EGENERIC;
    } else if (!chroma_compatible(fourcc, pic->format.i_chroma)
     || frame->width > (int) pic->format.i_width
     || frame->height > (int) pic->format.i_height)
    {
        msg_Warn(dec, "dropping frame because the vout changed");
        return VLC_EGENERIC;
    }

    for (int plane = 0; plane < pic->i_planes; plane++)
    {
        const uint8_t *src = frame->data[plane];
        uint8_t *dst = pic->p[plane].p_pixels;
        size_t src_stride = frame->linesize[plane];
        size_t dst_stride = pic->p[plane].i_pitch;
        size_t size = __MIN(src_stride, dst_stride);

        for (int line = 0; line < pic->p[plane].i_visible_lines; line++)
        {
            memcpy(dst, src, size);
            src += src_stride;
            dst += dst_stride;
        }
    }
    return VLC_SUCCESS;
}

static int OpenVideoCodec( decoder_t *p_dec )
{
    decoder_sys_t *p_sys = p_dec->p_sys;
    AVCodecContext *ctx = p_sys->p_context;
    const AVCodec *codec = p_sys->p_codec;
    int ret;

    if( ctx->extradata_size <= 0 )
    {
        if( codec->id == AV_CODEC_ID_VC1 ||
            codec->id == AV_CODEC_ID_THEORA )
        {
            msg_Warn( p_dec, "waiting for extra data for codec %s",
                      codec->name );
            return 1;
        }
    }

    ctx->width  = p_dec->fmt_in.video.i_visible_width;
    ctx->height = p_dec->fmt_in.video.i_visible_height;

    ctx->coded_width = p_dec->fmt_in.video.i_width;
    ctx->coded_height = p_dec->fmt_in.video.i_height;

    ctx->bits_per_coded_sample = p_dec->fmt_in.video.i_bits_per_pixel;
    p_sys->pix_fmt = AV_PIX_FMT_NONE;
    p_sys->profile = -1;
    p_sys->level = -1;
    cc_Init( &p_sys->cc );
