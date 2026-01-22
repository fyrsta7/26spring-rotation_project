        p_dec->fmt_out.video.i_sar_den = p_dec->fmt_in.video.i_sar_den;
    }
    else
    {
        p_dec->fmt_out.video.i_sar_num = p_context->sample_aspect_ratio.num;
        p_dec->fmt_out.video.i_sar_den = p_context->sample_aspect_ratio.den;

        if( !p_dec->fmt_out.video.i_sar_num || !p_dec->fmt_out.video.i_sar_den )
        {
            p_dec->fmt_out.video.i_sar_num = 1;
            p_dec->fmt_out.video.i_sar_den = 1;
        }
    }

    if( p_dec->fmt_in.video.i_frame_rate > 0 &&
        p_dec->fmt_in.video.i_frame_rate_base > 0 )
    {
        p_dec->fmt_out.video.i_frame_rate =
            p_dec->fmt_in.video.i_frame_rate;
        p_dec->fmt_out.video.i_frame_rate_base =
            p_dec->fmt_in.video.i_frame_rate_base;
    }
    else if( p_context->time_base.num > 0 && p_context->time_base.den > 0 )
    {
        p_dec->fmt_out.video.i_frame_rate = p_context->time_base.den;
        p_dec->fmt_out.video.i_frame_rate_base = p_context->time_base.num;
    }

    return decoder_NewPicture( p_dec );
}

/*****************************************************************************
 * InitVideo: initialize the video decoder
 *****************************************************************************
 * the ffmpeg codec will be opened, some memory allocated. The vout is not yet
 * opened (done after the first decoded frame).
 *****************************************************************************/
int InitVideoDec( decoder_t *p_dec, AVCodecContext *p_context,
                      AVCodec *p_codec, int i_codec_id, const char *psz_namecodec )
{
    decoder_sys_t *p_sys;
    int i_val;

    /* Allocate the memory needed to store the decoder's structure */
    if( ( p_dec->p_sys = p_sys = calloc( 1, sizeof(decoder_sys_t) ) ) == NULL )
        return VLC_ENOMEM;

    p_codec->type = AVMEDIA_TYPE_VIDEO;
    p_context->codec_type = AVMEDIA_TYPE_VIDEO;
    p_context->codec_id = i_codec_id;
    p_sys->p_context = p_context;
    p_sys->p_codec = p_codec;
    p_sys->i_codec_id = i_codec_id;
    p_sys->psz_namecodec = psz_namecodec;
    p_sys->p_ff_pic = avcodec_alloc_frame();
    p_sys->b_delayed_open = true;
    p_sys->p_va = NULL;
    vlc_sem_init( &p_sys->sem_mt, 0 );

    /* ***** Fill p_context with init values ***** */
    p_sys->p_context->codec_tag = ffmpeg_CodecTag( p_dec->fmt_in.i_original_fourcc ?: p_dec->fmt_in.i_codec );

    /*  ***** Get configuration of ffmpeg plugin ***** */
    p_sys->p_context->workaround_bugs =
        var_InheritInteger( p_dec, "ffmpeg-workaround-bugs" );
    p_sys->p_context->error_recognition =
        var_InheritInteger( p_dec, "ffmpeg-error-resilience" );

    if( var_CreateGetBool( p_dec, "grayscale" ) )
        p_sys->p_context->flags |= CODEC_FLAG_GRAY;

    i_val = var_CreateGetInteger( p_dec, "ffmpeg-vismv" );
    if( i_val ) p_sys->p_context->debug_mv = i_val;

    i_val = var_CreateGetInteger( p_dec, "ffmpeg-lowres" );
    if( i_val > 0 && i_val <= 2 ) p_sys->p_context->lowres = i_val;

    i_val = var_CreateGetInteger( p_dec, "ffmpeg-skiploopfilter" );
    if( i_val >= 4 ) p_sys->p_context->skip_loop_filter = AVDISCARD_ALL;
    else if( i_val == 3 ) p_sys->p_context->skip_loop_filter = AVDISCARD_NONKEY;
    else if( i_val == 2 ) p_sys->p_context->skip_loop_filter = AVDISCARD_BIDIR;
    else if( i_val == 1 ) p_sys->p_context->skip_loop_filter = AVDISCARD_NONREF;

    if( var_CreateGetBool( p_dec, "ffmpeg-fast" ) )
        p_sys->p_context->flags2 |= CODEC_FLAG2_FAST;

    /* ***** ffmpeg frame skipping ***** */
    p_sys->b_hurry_up = var_CreateGetBool( p_dec, "ffmpeg-hurry-up" );

    switch( var_CreateGetInteger( p_dec, "ffmpeg-skip-frame" ) )
    {
        case -1:
            p_sys->p_context->skip_frame = AVDISCARD_NONE;
            break;
        case 0:
            p_sys->p_context->skip_frame = AVDISCARD_DEFAULT;
            break;
        case 1:
            p_sys->p_context->skip_frame = AVDISCARD_NONREF;
            break;
        case 2:
            p_sys->p_context->skip_frame = AVDISCARD_NONKEY;
            break;
        case 3:
            p_sys->p_context->skip_frame = AVDISCARD_ALL;
            break;
        default:
            p_sys->p_context->skip_frame = AVDISCARD_NONE;
            break;
    }
    p_sys->i_skip_frame = p_sys->p_context->skip_frame;

    switch( var_CreateGetInteger( p_dec, "ffmpeg-skip-idct" ) )
    {
        case -1:
            p_sys->p_context->skip_idct = AVDISCARD_NONE;
            break;
        case 0:
            p_sys->p_context->skip_idct = AVDISCARD_DEFAULT;
            break;
        case 1:
            p_sys->p_context->skip_idct = AVDISCARD_NONREF;
            break;
        case 2:
            p_sys->p_context->skip_idct = AVDISCARD_NONKEY;
            break;
        case 3:
            p_sys->p_context->skip_idct = AVDISCARD_ALL;
            break;
        default:
            p_sys->p_context->skip_idct = AVDISCARD_NONE;
            break;
    }
    p_sys->i_skip_idct = p_sys->p_context->skip_idct;

    /* ***** ffmpeg direct rendering ***** */
    p_sys->b_direct_rendering = false;
    p_sys->i_direct_rendering_used = -1;
    if( var_CreateGetBool( p_dec, "ffmpeg-dr" ) &&
       (p_sys->p_codec->capabilities & CODEC_CAP_DR1) &&
        /* No idea why ... but this fixes flickering on some TSCC streams */
        p_sys->i_codec_id != CODEC_ID_TSCC && p_sys->i_codec_id != CODEC_ID_CSCD &&
#if (LIBAVCODEC_VERSION_INT >= AV_VERSION_INT( 52, 68, 2 ) ) && (LIBAVCODEC_VERSION_INT < AV_VERSION_INT( 52, 100, 1 ) )
        /* avcodec native vp8 decode doesn't handle EMU_EDGE flag, and I
           don't have idea howto implement fallback to libvpx decoder */
        p_sys->i_codec_id != CODEC_ID_VP8 &&
#endif
        !p_sys->p_context->debug_mv )
    {
        /* Some codecs set pix_fmt only after the 1st frame has been decoded,
         * so we need to do another check in ffmpeg_GetFrameBuf() */
        p_sys->b_direct_rendering = true;
    }

    /* ffmpeg doesn't properly release old pictures when frames are skipped */
    //if( p_sys->b_hurry_up ) p_sys->b_direct_rendering = false;
    if( p_sys->b_direct_rendering )
    {
        msg_Dbg( p_dec, "trying to use direct rendering" );
        p_sys->p_context->flags |= CODEC_FLAG_EMU_EDGE;
    }
    else
    {
        msg_Dbg( p_dec, "direct rendering is disabled" );
    }

    /* Always use our get_buffer wrapper so we can calculate the
     * PTS correctly */
    p_sys->p_context->get_buffer = ffmpeg_GetFrameBuf;
    p_sys->p_context->reget_buffer = ffmpeg_ReGetFrameBuf;
    p_sys->p_context->release_buffer = ffmpeg_ReleaseFrameBuf;
    p_sys->p_context->opaque = p_dec;

#ifdef HAVE_AVCODEC_MT
    int i_thread_count = var_InheritInteger( p_dec, "ffmpeg-threads" );
    if( i_thread_count <= 0 )
        i_thread_count = vlc_GetCPUCount();
    i_thread_count = __MIN( i_thread_count, 16 );
    msg_Dbg( p_dec, "allowing %d thread(s) for decoding", i_thread_count );
    p_sys->p_context->thread_count = i_thread_count;
#endif

#ifdef HAVE_AVCODEC_VA
    const bool b_use_hw = var_CreateGetBool( p_dec, "ffmpeg-hw" );
    if( b_use_hw &&
        (i_codec_id == CODEC_ID_MPEG1VIDEO || i_codec_id == CODEC_ID_MPEG2VIDEO ||
         i_codec_id == CODEC_ID_MPEG4 ||
         i_codec_id == CODEC_ID_H264 ||
         i_codec_id == CODEC_ID_VC1 || i_codec_id == CODEC_ID_WMV3) )
    {
#ifdef HAVE_AVCODEC_MT
        if( p_sys->p_context->thread_type & FF_THREAD_FRAME )
        {
            msg_Warn( p_dec, "threaded frame decoding is not compatible with ffmpeg-hw, disabled" );
            p_sys->p_context->thread_type &= ~FF_THREAD_FRAME;
        }
#endif
        p_sys->p_context->get_format = ffmpeg_GetFormat;
    }
#endif
#ifdef HAVE_AVCODEC_MT
    if( p_sys->p_context->thread_type & FF_THREAD_FRAME )
        p_dec->i_extra_picture_buffers = 2 * p_sys->p_context->thread_count;
#endif


    /* ***** misc init ***** */
    p_sys->i_pts = VLC_TS_INVALID;
    p_sys->b_has_b_frames = false;
    p_sys->b_first_frame = true;
    p_sys->b_flush = false;
    p_sys->i_late_frames = 0;

    /* Set output properties */
    p_dec->fmt_out.i_cat = VIDEO_ES;
    if( GetVlcChroma( &p_dec->fmt_out.video, p_context->pix_fmt ) != VLC_SUCCESS )
    {
        /* we are doomed. but not really, because most codecs set their pix_fmt later on */
        p_dec->fmt_out.i_codec = VLC_CODEC_I420;
    }
    p_dec->fmt_out.i_codec = p_dec->fmt_out.video.i_chroma;

    /* Setup palette */
    memset( &p_sys->palette, 0, sizeof(p_sys->palette) );
    if( p_dec->fmt_in.video.p_palette )
    {
        p_sys->palette.palette_changed = 1;
