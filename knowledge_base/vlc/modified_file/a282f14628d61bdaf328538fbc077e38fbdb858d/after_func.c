 * Local prototypes
 *****************************************************************************/
static int  OpenDecoder( vlc_object_t * );
static void CloseDecoder( vlc_object_t * );

static picture_t *DecodeBlock( decoder_t *, block_t ** );
#if MPEG2_RELEASE >= MPEG2_VERSION (0, 5, 0)
static block_t   *GetCc( decoder_t *p_dec, bool pb_present[4] );
#endif

static picture_t *GetNewPicture( decoder_t * );
static void PutPicture( decoder_t *, picture_t * );

static void GetAR( decoder_t *p_dec );

static void Reset( decoder_t *p_dec );

/* */
static void DpbInit( decoder_t * );
static void DpbClean( decoder_t * );
static picture_t *DpbNewPicture( decoder_t * );
static void DpbUnlinkPicture( decoder_t *, picture_t * );
static int DpbDisplayPicture( decoder_t *, picture_t * );

/*****************************************************************************
 * Module descriptor
 *****************************************************************************/
vlc_module_begin ()
    set_description( N_("MPEG I/II video decoder (using libmpeg2)") )
    set_capability( "decoder", 150 )
    set_category( CAT_INPUT )
    set_subcategory( SUBCAT_INPUT_VCODEC )
    set_callbacks( OpenDecoder, CloseDecoder )
    add_shortcut( "libmpeg2" )
vlc_module_end ()

/*****************************************************************************
 * OpenDecoder: probe the decoder and return score
 *****************************************************************************/
static int OpenDecoder( vlc_object_t *p_this )
{
    decoder_t *p_dec = (decoder_t*)p_this;
    decoder_sys_t *p_sys;
    uint32_t i_accel = 0;

    if( p_dec->fmt_in.i_codec != VLC_CODEC_MPGV )
        return VLC_EGENERIC;

    /* Select onl recognized original format (standard mpeg video) */
    switch( p_dec->fmt_in.i_original_fourcc )
    {
    case VLC_FOURCC('m','p','g','1'):
    case VLC_FOURCC('m','p','g','2'):
    case VLC_FOURCC('m','p','g','v'):
    case VLC_FOURCC('P','I','M','1'):
    case VLC_FOURCC('h','d','v','2'):
        break;
    default:
        if( p_dec->fmt_in.i_original_fourcc )
            return VLC_EGENERIC;
        break;
    }

    /* Allocate the memory needed to store the decoder's structure */
    if( ( p_dec->p_sys = p_sys = calloc( 1, sizeof(*p_sys)) ) == NULL )
        return VLC_ENOMEM;

    /* Initialize the thread properties */
    p_sys->p_mpeg2dec = NULL;
    p_sys->p_synchro  = NULL;
    p_sys->p_info     = NULL;
    p_sys->i_current_pts  = 0;
    p_sys->i_previous_pts = 0;
    p_sys->i_current_dts  = 0;
    p_sys->i_previous_dts = 0;
    p_sys->i_aspect = 0;
    p_sys->b_garbage_pic = false;
    p_sys->b_slice_i  = false;
    p_sys->b_second_field = false;
    p_sys->b_skip     = false;
    p_sys->b_preroll = false;
    DpbInit( p_dec );

    p_sys->i_cc_pts = 0;
    p_sys->i_cc_dts = 0;
    p_sys->i_cc_flags = 0;
#if MPEG2_RELEASE >= MPEG2_VERSION (0, 5, 0)
    p_dec->pf_get_cc = GetCc;
    cc_Init( &p_sys->cc );
#endif
    p_sys->p_gop_user_data = NULL;
    p_sys->i_gop_user_data = 0;

#if defined( __i386__ ) || defined( __x86_64__ )
    if( vlc_CPU() & CPU_CAPABILITY_MMX )
    {
        i_accel |= MPEG2_ACCEL_X86_MMX;
    }

    if( vlc_CPU() & CPU_CAPABILITY_3DNOW )
    {
        i_accel |= MPEG2_ACCEL_X86_3DNOW;
    }

    if( vlc_CPU() & CPU_CAPABILITY_MMXEXT )
    {
        i_accel |= MPEG2_ACCEL_X86_MMXEXT;
    }

#elif defined( __powerpc__ ) || defined( __ppc__ ) || defined( __ppc64__ )
