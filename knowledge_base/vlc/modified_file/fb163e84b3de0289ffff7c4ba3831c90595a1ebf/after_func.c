    "8x8dct", "analyse", "asm", "aud", "bframes", "bime", "bpyramid",
    "b-adapt", "b-bias", "b-rdo", "cabac", "chroma-me", "chroma-qp-offset",
    "cplxblur", "crf", "dct-decimate", "deadzone-inter", "deadzone-intra",
    "deblock", "direct", "direct-8x8", "filter", "fast-pskip", "frameref",
    "interlaced", "ipratio", "keyint", "keyint-min", "level", "loopfilter",
    "me", "merange", "min-keyint", "mixed-refs", "mvrange", "mvrange-thread",
    "nf", "non-deterministic", "nr", "partitions", "pass", "pbratio",
    "pre-scenecut", "psnr", "qblur", "qp", "qcomp", "qpstep", "qpmax",
    "qpmin", "qp-max", "qp-min", "quiet", "ratetol", "ref", "scenecut",
    "sps-id", "ssim", "stats", "subme", "subpel", "tolerance", "trellis",
    "verbose", "vbv-bufsize", "vbv-init", "vbv-maxrate", "weightb", NULL
};

static block_t *Encode( encoder_t *, picture_t * );

struct encoder_sys_t
{
    x264_t          *h;
    x264_param_t    param;

    int             i_buffer;
    uint8_t         *p_buffer;

    mtime_t         i_interpolated_dts;

    char *psz_stat_name;
};

/*****************************************************************************
 * Open: probe the encoder
 *****************************************************************************/
static int  Open ( vlc_object_t *p_this )
{
    encoder_t     *p_enc = (encoder_t *)p_this;
    encoder_sys_t *p_sys;
    vlc_value_t    val;
    int i_qmin = 0, i_qmax = 0;
    x264_nal_t    *nal;
    int i, i_nal;

    if( p_enc->fmt_out.i_codec != VLC_FOURCC( 'h', '2', '6', '4' ) &&
        !p_enc->b_force )
    {
        return VLC_EGENERIC;
    }

    /* X264_POINTVER or X264_VERSION are not available */
    msg_Dbg ( p_enc, "version x264 0.%d.X", X264_BUILD );

#if X264_BUILD < 37
    if( p_enc->fmt_in.video.i_width % 16 != 0 ||
        p_enc->fmt_in.video.i_height % 16 != 0 )
    {
        msg_Warn( p_enc, "size is not a multiple of 16 (%ix%i)",
                  p_enc->fmt_in.video.i_width, p_enc->fmt_in.video.i_height );

        if( p_enc->fmt_in.video.i_width < 16 ||
            p_enc->fmt_in.video.i_height < 16 )
        {
            msg_Err( p_enc, "video is too small to be cropped" );
            return VLC_EGENERIC;
        }

        msg_Warn( p_enc, "cropping video to %ix%i",
                  p_enc->fmt_in.video.i_width >> 4 << 4,
                  p_enc->fmt_in.video.i_height >> 4 << 4 );
    }
#endif

    config_ChainParse( p_enc, SOUT_CFG_PREFIX, ppsz_sout_options, p_enc->p_cfg );

    p_enc->fmt_out.i_codec = VLC_FOURCC( 'h', '2', '6', '4' );
    p_enc->fmt_in.i_codec = VLC_FOURCC('I','4','2','0');

    p_enc->pf_encode_video = Encode;
    p_enc->pf_encode_audio = NULL;
    p_enc->p_sys = p_sys = malloc( sizeof( encoder_sys_t ) );
    p_sys->i_interpolated_dts = 0;
    p_sys->psz_stat_name = NULL;

    x264_param_default( &p_sys->param );
    p_sys->param.i_width  = p_enc->fmt_in.video.i_width;
    p_sys->param.i_height = p_enc->fmt_in.video.i_height;
#if X264_BUILD < 37
    p_sys->param.i_width  = p_sys->param.i_width >> 4 << 4;
    p_sys->param.i_height = p_sys->param.i_height >> 4 << 4;
#endif

    /* average bitrate specified by transcode vb */
    p_sys->param.rc.i_bitrate = p_enc->fmt_out.i_bitrate / 1000;

#if X264_BUILD < 48
    /* cbr = 1 overrides qp or crf and sets an average bitrate
       but maxrate = average bitrate is needed for "real" CBR */
    if( p_sys->param.rc.i_bitrate > 0 ) p_sys->param.rc.b_cbr = 1;
#else
    if( p_sys->param.rc.i_bitrate > 0 ) p_sys->param.rc.i_rc_method = X264_RC_ABR;
#endif

    var_Get( p_enc, SOUT_CFG_PREFIX "qpstep", &val );
    if( val.i_int >= 0 && val.i_int <= 51 ) p_sys->param.rc.i_qp_step = val.i_int;
    var_Get( p_enc, SOUT_CFG_PREFIX "qpmin", &val );
    if( val.i_int >= 0 && val.i_int <= 51 ) i_qmin = val.i_int;
    var_Get( p_enc, SOUT_CFG_PREFIX "qpmax", &val );
    if( val.i_int >= 0 && val.i_int <= 51 ) i_qmax = val.i_int;

    var_Get( p_enc, SOUT_CFG_PREFIX "qp", &val );
    if( val.i_int >= 0 && val.i_int <= 51 )
    {
        if( i_qmin > val.i_int ) i_qmin = val.i_int;
        if( i_qmax < val.i_int ) i_qmax = val.i_int;

#if X264_BUILD >= 0x000a
        p_sys->param.rc.i_qp_constant = val.i_int;
        p_sys->param.rc.i_qp_min = i_qmin;
        p_sys->param.rc.i_qp_max = i_qmax;
#else
        p_sys->param.i_qp_constant = val.i_int;
#endif
    }

#if X264_BUILD >= 24
    var_Get( p_enc, SOUT_CFG_PREFIX "ratetol", &val );
    p_sys->param.rc.f_rate_tolerance = val.f_float;

    var_Get( p_enc, SOUT_CFG_PREFIX "vbv-init", &val );
    p_sys->param.rc.f_vbv_buffer_init = val.f_float;

    var_Get( p_enc, SOUT_CFG_PREFIX "vbv-bufsize", &val );
    p_sys->param.rc.i_vbv_buffer_size = val.i_int;

    /* x264 vbv-bufsize = 0 (default). if not provided set period
       in seconds for local maximum bitrate (cache/bufsize) based
       on average bitrate. */
    if( !val.i_int )
        p_sys->param.rc.i_vbv_buffer_size = p_sys->param.rc.i_bitrate * 2;

    /* max bitrate = average bitrate -> CBR */
    var_Get( p_enc, SOUT_CFG_PREFIX "vbv-maxrate", &val );
    p_sys->param.rc.i_vbv_max_bitrate = val.i_int;

#else
    p_sys->param.rc.i_rc_buffer_size = p_sys->param.rc.i_bitrate;
    p_sys->param.rc.i_rc_init_buffer = p_sys->param.rc.i_bitrate / 4;
#endif

    var_Get( p_enc, SOUT_CFG_PREFIX "cabac", &val );
    p_sys->param.b_cabac = val.b_bool;

    /* disable deblocking when nf (no loop filter) is enabled */
    var_Get( p_enc, SOUT_CFG_PREFIX "nf", &val );
    p_sys->param.b_deblocking_filter = !val.b_bool;

    var_Get( p_enc, SOUT_CFG_PREFIX "deblock", &val );
    if( val.psz_string )
    {
        char *p = strchr( val.psz_string, ':' );
        p_sys->param.i_deblocking_filter_alphac0 = atoi( val.psz_string );
        p_sys->param.i_deblocking_filter_beta = p ? atoi( p+1 ) : p_sys->param.i_deblocking_filter_alphac0;
        free( val.psz_string );
    }

    var_Get( p_enc, SOUT_CFG_PREFIX "level", &val );
    if( val.psz_string )
    {
        if( atof (val.psz_string) < 6 )
            p_sys->param.i_level_idc = (int) ( 10 * atof (val.psz_string) + .5);
        else
            p_sys->param.i_level_idc = atoi (val.psz_string);
        free( val.psz_string );
    }

#if X264_BUILD >= 51 /* r570 */
    var_Get( p_enc, SOUT_CFG_PREFIX "interlaced", &val );
    p_sys->param.b_interlaced = val.b_bool;
#endif

    var_Get( p_enc, SOUT_CFG_PREFIX "ipratio", &val );
    p_sys->param.rc.f_ip_factor = val.f_float;

    var_Get( p_enc, SOUT_CFG_PREFIX "pbratio", &val );
    p_sys->param.rc.f_pb_factor = val.f_float;

    var_Get( p_enc, SOUT_CFG_PREFIX "qcomp", &val );
    p_sys->param.rc.f_qcompress = val.f_float;

    var_Get( p_enc, SOUT_CFG_PREFIX "cplxblur", &val );
    p_sys->param.rc.f_complexity_blur = val.f_float;

    var_Get( p_enc, SOUT_CFG_PREFIX "qblur", &val );
    p_sys->param.rc.f_qblur = val.f_float;

#if X264_BUILD >= 0x000e
    var_Get( p_enc, SOUT_CFG_PREFIX "verbose", &val );
    if( val.b_bool ) p_sys->param.i_log_level = X264_LOG_DEBUG;
#endif

    var_Get( p_enc, SOUT_CFG_PREFIX "quiet", &val );
    if( val.b_bool ) p_sys->param.i_log_level = X264_LOG_NONE;

#if X264_BUILD >= 47 /* r518 */
    var_Get( p_enc, SOUT_CFG_PREFIX "sps-id", &val );
    if( val.i_int >= 0 ) p_sys->param.i_sps_id = val.i_int;
#endif

    var_Get( p_enc, SOUT_CFG_PREFIX "aud", &val );
    if( val.b_bool ) p_sys->param.b_aud = val.b_bool;

    var_Get( p_enc, SOUT_CFG_PREFIX "keyint", &val );
#if X264_BUILD >= 0x000e
    if( val.i_int > 0 ) p_sys->param.i_keyint_max = val.i_int;
#else
    if( val.i_int > 0 ) p_sys->param.i_iframe = val.i_int;
#endif

    var_Get( p_enc, SOUT_CFG_PREFIX "min-keyint", &val );
#if X264_BUILD >= 0x000e
    if( val.i_int > 0 ) p_sys->param.i_keyint_min = val.i_int;
#else
    if( val.i_int > 0 ) p_sys->param.i_idrframe = val.i_int;
#endif

    var_Get( p_enc, SOUT_CFG_PREFIX "bframes", &val );
    if( val.i_int >= 0 && val.i_int <= 16 )
        p_sys->param.i_bframe = val.i_int;

#if X264_BUILD >= 22
    var_Get( p_enc, SOUT_CFG_PREFIX "bpyramid", &val );
    p_sys->param.b_bframe_pyramid = val.b_bool;
#endif

    var_Get( p_enc, SOUT_CFG_PREFIX "ref", &val );
    if( val.i_int > 0 && val.i_int <= 15 )
        p_sys->param.i_frame_reference = val.i_int;

    var_Get( p_enc, SOUT_CFG_PREFIX "scenecut", &val );
#if X264_BUILD >= 0x000b
    if( val.i_int >= -1 && val.i_int <= 100 )
        p_sys->param.i_scenecut_threshold = val.i_int;
#endif

#if X264_BUILD >= 55 /* r607 */
    var_Get( p_enc, SOUT_CFG_PREFIX "pre-scenecut", &val );
    p_sys->param.b_pre_scenecut = val.b_bool;
    var_Get( p_enc, SOUT_CFG_PREFIX "non-deterministic", &val );
    p_sys->param.b_deterministic = val.b_bool;
#endif

    var_Get( p_enc, SOUT_CFG_PREFIX "subme", &val );
    if( val.i_int >= 1 && val.i_int <= SUBME_MAX )
        p_sys->param.analyse.i_subpel_refine = val.i_int;

#if X264_BUILD >= 24
    var_Get( p_enc, SOUT_CFG_PREFIX "me", &val );
    if( !strcmp( val.psz_string, "dia" ) )
    {
        p_sys->param.analyse.i_me_method = X264_ME_DIA;
    }
    else if( !strcmp( val.psz_string, "hex" ) )
    {
        p_sys->param.analyse.i_me_method = X264_ME_HEX;
    }
    else if( !strcmp( val.psz_string, "umh" ) )
    {
        p_sys->param.analyse.i_me_method = X264_ME_UMH;
    }
    else if( !strcmp( val.psz_string, "esa" ) )
    {
        p_sys->param.analyse.i_me_method = X264_ME_ESA;
    }
    #if X264_BUILD >= 58 /* r728 */
        else if( !strcmp( val.psz_string, "tesa" ) )
        {
            p_sys->param.analyse.i_me_method = X264_ME_TESA;
        }
    #endif
    if( val.psz_string ) free( val.psz_string );

    var_Get( p_enc, SOUT_CFG_PREFIX "merange", &val );
    if( val.i_int >= 0 && val.i_int <= 64 )
        p_sys->param.analyse.i_me_range = val.i_int;

    var_Get( p_enc, SOUT_CFG_PREFIX "mvrange", &val );
        p_sys->param.analyse.i_mv_range = val.i_int;
#endif

#if X264_BUILD >= 55 /* r607 */
    var_Get( p_enc, SOUT_CFG_PREFIX "mvrange-thread", &val );
        p_sys->param.analyse.i_mv_range_thread = val.i_int;
#endif

    var_Get( p_enc, SOUT_CFG_PREFIX "direct", &val );
    if( !strcmp( val.psz_string, "none" ) )
    {
        p_sys->param.analyse.i_direct_mv_pred = X264_DIRECT_PRED_NONE;
    }
    else if( !strcmp( val.psz_string, "spatial" ) )
    {
        p_sys->param.analyse.i_direct_mv_pred = X264_DIRECT_PRED_SPATIAL;
    }
    else if( !strcmp( val.psz_string, "temporal" ) )
    {
        p_sys->param.analyse.i_direct_mv_pred = X264_DIRECT_PRED_TEMPORAL;
    }
#if X264_BUILD >= 45 /* r457 */
    else if( !strcmp( val.psz_string, "auto" ) )
    {
        p_sys->param.analyse.i_direct_mv_pred = X264_DIRECT_PRED_AUTO;
    }
#endif
    if( val.psz_string ) free( val.psz_string );

    var_Get( p_enc, SOUT_CFG_PREFIX "psnr", &val );
    p_sys->param.analyse.b_psnr = val.b_bool;

#if X264_BUILD >= 50 /* r554 */
    var_Get( p_enc, SOUT_CFG_PREFIX "ssim", &val );
    p_sys->param.analyse.b_ssim = val.b_bool;
#endif

#if X264_BUILD >= 0x0012
    var_Get( p_enc, SOUT_CFG_PREFIX "weightb", &val );
    p_sys->param.analyse.b_weighted_bipred = val.b_bool;
#endif

#if X264_BUILD >= 0x0013
    var_Get( p_enc, SOUT_CFG_PREFIX "b-adapt", &val );
    p_sys->param.b_bframe_adaptive = val.b_bool;

    var_Get( p_enc, SOUT_CFG_PREFIX "b-bias", &val );
    if( val.i_int >= -100 && val.i_int <= 100 )
        p_sys->param.i_bframe_bias = val.i_int;
#endif

#if X264_BUILD >= 23
    var_Get( p_enc, SOUT_CFG_PREFIX "chroma-me", &val );
    p_sys->param.analyse.b_chroma_me = val.b_bool;
    var_Get( p_enc, SOUT_CFG_PREFIX "chroma-qp-offset", &val );
    p_sys->param.analyse.i_chroma_qp_offset = val.i_int;
#endif

#if X264_BUILD >= 36
    var_Get( p_enc, SOUT_CFG_PREFIX "mixed-refs", &val );
    p_sys->param.analyse.b_mixed_references = val.b_bool;
#endif

#if X264_BUILD >= 37
    var_Get( p_enc, SOUT_CFG_PREFIX "crf", &val );
    if( val.i_int > 0 && val.i_int <= 51 )
    {
#if X264_BUILD >= 54
        p_sys->param.rc.f_rf_constant = val.i_int;
#else
        p_sys->param.rc.i_rf_constant = val.i_int;
#endif
#if X264_BUILD >= 48
        p_sys->param.rc.i_rc_method = X264_RC_CRF;
#endif
    }
#endif

#if X264_BUILD >= 39
    var_Get( p_enc, SOUT_CFG_PREFIX "trellis", &val );
    if( val.i_int >= 0 && val.i_int <= 2 )
        p_sys->param.analyse.i_trellis = val.i_int;
#endif

#if X264_BUILD >= 41
    var_Get( p_enc, SOUT_CFG_PREFIX "b-rdo", &val );
    p_sys->param.analyse.b_bframe_rdo = val.b_bool;
#endif

#if X264_BUILD >= 42
    var_Get( p_enc, SOUT_CFG_PREFIX "fast-pskip", &val );
    p_sys->param.analyse.b_fast_pskip = val.b_bool;
#endif

#if X264_BUILD >= 43
    var_Get( p_enc, SOUT_CFG_PREFIX "bime", &val );
    p_sys->param.analyse.b_bidir_me = val.b_bool;
#endif

#if X264_BUILD >= 44
    var_Get( p_enc, SOUT_CFG_PREFIX "nr", &val );
    if( val.i_int >= 0 && val.i_int <= 1000 )
        p_sys->param.analyse.i_noise_reduction = val.i_int;
#endif

#if X264_BUILD >= 46
    var_Get( p_enc, SOUT_CFG_PREFIX "dct-decimate", &val );
    p_sys->param.analyse.b_dct_decimate = val.b_bool;
#endif

#if X264_BUILD >= 52
    var_Get( p_enc, SOUT_CFG_PREFIX "deadzone-inter", &val );
    if( val.i_int >= 0 && val.i_int <= 32 )
        p_sys->param.analyse.i_luma_deadzone[0] = val.i_int;

    var_Get( p_enc, SOUT_CFG_PREFIX "deadzone-intra", &val );
    if( val.i_int >= 0 && val.i_int <= 32 )
        p_sys->param.analyse.i_luma_deadzone[1] = val.i_int;

    var_Get( p_enc, SOUT_CFG_PREFIX "direct-8x8", &val );
    if( val.i_int >= -1 && val.i_int <= 1 )
        p_sys->param.analyse.i_direct_8x8_inference = val.i_int;
#endif

    var_Get( p_enc, SOUT_CFG_PREFIX "asm", &val );
    if( !val.b_bool ) p_sys->param.cpu = 0;

#ifndef X264_ANALYSE_BSUB16x16
#   define X264_ANALYSE_BSUB16x16 0
#endif
    var_Get( p_enc, SOUT_CFG_PREFIX "partitions", &val );
    if( !strcmp( val.psz_string, "none" ) )
    {
        p_sys->param.analyse.inter = 0;
    }
    else if( !strcmp( val.psz_string, "fast" ) )
    {
        p_sys->param.analyse.inter = X264_ANALYSE_I4x4;
    }
    else if( !strcmp( val.psz_string, "normal" ) )
    {
        p_sys->param.analyse.inter =
            X264_ANALYSE_I4x4 |
            X264_ANALYSE_PSUB16x16;
#ifdef X264_ANALYSE_I8x8
        p_sys->param.analyse.inter |= X264_ANALYSE_I8x8;
#endif
    }
    else if( !strcmp( val.psz_string, "slow" ) )
    {
        p_sys->param.analyse.inter =
            X264_ANALYSE_I4x4 |
            X264_ANALYSE_PSUB16x16 |
            X264_ANALYSE_BSUB16x16;
#ifdef X264_ANALYSE_I8x8
        p_sys->param.analyse.inter |= X264_ANALYSE_I8x8;
#endif
    }
    else if( !strcmp( val.psz_string, "all" ) )
    {
        p_sys->param.analyse.inter =
            X264_ANALYSE_I4x4 |
            X264_ANALYSE_PSUB16x16 |
            X264_ANALYSE_BSUB16x16 |
            X264_ANALYSE_PSUB8x8;
#ifdef X264_ANALYSE_I8x8
        p_sys->param.analyse.inter |= X264_ANALYSE_I8x8;
#endif
    }
    if( val.psz_string ) free( val.psz_string );

#if X264_BUILD >= 30
    var_Get( p_enc, SOUT_CFG_PREFIX "8x8dct", &val );
    p_sys->param.analyse.b_transform_8x8 = val.b_bool;
#endif

    if( p_enc->fmt_in.video.i_aspect > 0 )
    {
        int64_t i_num, i_den;
        unsigned int i_dst_num, i_dst_den;

        i_num = p_enc->fmt_in.video.i_aspect *
            (int64_t)p_enc->fmt_in.video.i_height;
        i_den = VOUT_ASPECT_FACTOR * p_enc->fmt_in.video.i_width;
        vlc_ureduce( &i_dst_num, &i_dst_den, i_num, i_den, 0 );

        p_sys->param.vui.i_sar_width = i_dst_num;
        p_sys->param.vui.i_sar_height = i_dst_den;
    }
    if( p_enc->fmt_in.video.i_frame_rate_base > 0 )
    {
        p_sys->param.i_fps_num = p_enc->fmt_in.video.i_frame_rate;
        p_sys->param.i_fps_den = p_enc->fmt_in.video.i_frame_rate_base;
    }

    unsigned i_cpu = vlc_CPU();
    if( !(i_cpu & CPU_CAPABILITY_MMX) )
    {
        p_sys->param.cpu &= ~X264_CPU_MMX;
    }
    if( !(i_cpu & CPU_CAPABILITY_MMXEXT) )
    {
        p_sys->param.cpu &= ~X264_CPU_MMXEXT;
    }
    if( !(i_cpu & CPU_CAPABILITY_SSE) )
    {
        p_sys->param.cpu &= ~X264_CPU_SSE;
    }
    if( !(i_cpu & CPU_CAPABILITY_SSE2) )
    {
        p_sys->param.cpu &= ~X264_CPU_SSE2;
    }

    /* BUILD 29 adds support for multi-threaded encoding while BUILD 49 (r543)
       also adds support for threads = 0 for automatically selecting an optimal
       value (cores * 1.5) based on detected CPUs. Default behavior for x264 is
       threads = 1, however VLC usage differs and uses threads = 0 (auto) by
       default unless ofcourse transcode threads is explicitly specified.. */
#if X264_BUILD >= 29
    p_sys->param.i_threads = p_enc->i_threads;
#endif

    var_Get( p_enc, SOUT_CFG_PREFIX "stats", &val );
    if( val.psz_string )
    {
        p_sys->param.rc.psz_stat_in  =
        p_sys->param.rc.psz_stat_out =
        p_sys->psz_stat_name         = val.psz_string;
    }

    var_Get( p_enc, SOUT_CFG_PREFIX "pass", &val );
    if( val.i_int > 0 && val.i_int <= 3 )
    {
        p_sys->param.rc.b_stat_write = val.i_int & 1;
        p_sys->param.rc.b_stat_read = val.i_int & 2;
    }

    /* We need to initialize pthreadw32 before we open the encoder,
       but only oncce for the whole application. Since pthreadw32
       doesn't keep a refcount, do it ourselves. */
#ifdef PTW32_STATIC_LIB
    vlc_value_t lock, count;

    var_Create( p_enc->p_libvlc, "pthread_win32_mutex", VLC_VAR_MUTEX );
    var_Get( p_enc->p_libvlc, "pthread_win32_mutex", &lock );
    vlc_mutex_lock( lock.p_address );

    var_Create( p_enc->p_libvlc, "pthread_win32_count", VLC_VAR_INTEGER );
    var_Get( p_enc->p_libvlc, "pthread_win32_count", &count );

    if( count.i_int == 0 )
    {   
        msg_Dbg( p_enc, "initializing pthread-win32" );
        if( !pthread_win32_process_attach_np() || !pthread_win32_thread_attach_np() )   
        {   
            msg_Warn( p_enc, "pthread Win32 Initialization failed" );
