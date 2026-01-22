static int  OpenDisplay ( vout_display_t *, video_format_t * );
static void CloseDisplay( vout_display_t * );

static int  KVALock( picture_t * );
static void KVAUnlock( picture_t * );

static void             MorphToPM     ( void );
static int              ConvertKey    ( USHORT );
static MRESULT EXPENTRY MyFrameWndProc( HWND, ULONG, MPARAM, MPARAM );
static MRESULT EXPENTRY WndProc       ( HWND, ULONG, MPARAM, MPARAM );

#define WC_VLC_KVA "WC_VLC_KVA"

#define COLOR_KEY 0x0F0F0F

#define WM_VLC_MANAGE               ( WM_USER + 1 )
#define WM_VLC_FULLSCREEN_CHANGE    ( WM_USER + 2 )
#define WM_VLC_SIZE_CHANGE          ( WM_USER + 3 )

#define TID_HIDE_MOUSE  0x1010

static const char *psz_video_mode[ 4 ] = {"DIVE", "WarpOverlay!", "SNAP",
                                          "VMAN"};

struct open_init
{
    vout_display_t *vd;
    const vout_display_cfg_t *cfg;
    video_format_t *fmtp;
};

static void PMThread( void *arg )
{
    struct open_init *init = ( struct open_init * )arg;
    vout_display_t *vd = init->vd;
    vout_display_sys_t * sys = vd->sys;
    const vout_display_cfg_t * cfg = init->cfg;
    video_format_t *fmtp = init->fmtp;
    ULONG i_frame_flags;
    QMSG qm;
    char *psz_mode;
    ULONG i_kva_mode;

    /* */
    video_format_t fmt;
    video_format_ApplyRotation(&fmt, fmtp);

    /* */
    vout_display_info_t info = vd->info;
    info.is_slow = false;
    info.has_double_click = true;
    info.has_pictures_invalid = false;

    MorphToPM();

    sys->hab = WinInitialize( 0 );
    sys->hmq = WinCreateMsgQueue( sys->hab, 0);

    WinRegisterClass( sys->hab,
                      WC_VLC_KVA,
                      WndProc,
                      CS_SIZEREDRAW | CS_MOVENOTIFY,
                      sizeof( PVOID ));

    sys->b_fixt23 = var_CreateGetBool( vd, "kva-fixt23");

    if( !sys->b_fixt23 )
        /* If an external window was specified, we'll draw in it. */
        sys->parent_window =
            vout_display_NewWindow( vd, VOUT_WINDOW_TYPE_HWND );

    if( sys->parent_window )
    {
        sys->parent = ( HWND )sys->parent_window->handle.hwnd;

        ULONG i_style = WinQueryWindowULong( sys->parent, QWL_STYLE );
        WinSetWindowULong( sys->parent, QWL_STYLE,
                           i_style | WS_CLIPCHILDREN );

        i_frame_flags = FCF_TITLEBAR;
    }
    else
    {
        sys->parent = HWND_DESKTOP;

        i_frame_flags = FCF_SYSMENU    | FCF_TITLEBAR | FCF_MINMAX |
                        FCF_SIZEBORDER | FCF_TASKLIST;
    }

    sys->frame =
        WinCreateStdWindow( sys->parent,      /* parent window handle */
                            WS_VISIBLE,       /* frame window style */
                            &i_frame_flags,   /* window style */
                            WC_VLC_KVA,       /* class name */
                            "",               /* window title */
                            0L,               /* default client style */
                            NULLHANDLE,       /* resource in exe file */
                            1,                /* frame window id */
                            &sys->client );   /* client window handle */

    if( sys->frame == NULLHANDLE )
    {
        msg_Err( vd, "cannot create a frame window");

        goto exit_frame;
    }

    WinSetWindowPtr( sys->client, 0, vd );

    if( !sys->parent_window )
    {
        WinSetWindowPtr( sys->frame, 0, vd );
        sys->p_old_frame = WinSubclassWindow( sys->frame, MyFrameWndProc );
    }

    psz_mode = var_CreateGetString( vd, "kva-video-mode" );

    i_kva_mode = KVAM_AUTO;
    if( strcmp( psz_mode, "snap" ) == 0 )
        i_kva_mode = KVAM_SNAP;
    else if( strcmp( psz_mode, "wo" ) == 0 )
        i_kva_mode = KVAM_WO;
    else if( strcmp( psz_mode, "vman" ) == 0 )
        i_kva_mode = KVAM_VMAN;
    else if( strcmp( psz_mode, "dive" ) == 0 )
        i_kva_mode = KVAM_DIVE;

    free( psz_mode );

    if( kvaInit( i_kva_mode, sys->client, COLOR_KEY ))
    {
        msg_Err( vd, "cannot initialize KVA");

        goto exit_kva_init;
    }

    kvaCaps( &sys->kvac );

    msg_Dbg( vd, "selected video mode = %s",
             psz_video_mode[ sys->kvac.ulMode - 1 ]);

    if( OpenDisplay( vd, &fmt ) )
    {
        msg_Err( vd, "cannot open display");

        goto exit_open_display;
    }

    if( cfg->is_fullscreen && !sys->parent_window )
        WinPostMsg( sys->client, WM_VLC_FULLSCREEN_CHANGE,
                    MPFROMLONG( true ), 0 );

    kvaDisableScreenSaver();

    /* Setup vout_display now that everything is fine */
    *fmtp       = fmt;
    vd->info    = info;

    vd->pool    = Pool;
    vd->prepare = NULL;
    vd->display = Display;
    vd->control = Control;

    /* Prevent SIG_FPE */
    _control87(MCW_EM, MCW_EM);

    sys->i_result = VLC_SUCCESS;
    DosPostEventSem( sys->ack_event );

    if( !sys->parent_window )
        WinSetVisibleRegionNotify( sys->frame, TRUE );

