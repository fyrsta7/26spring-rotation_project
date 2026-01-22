    bool            can_blit_fourcc;

    uint32_t        i_rgb_colorkey;      /* colorkey in RGB used by the overlay */
    uint32_t        i_colorkey;                 /* colorkey used by the overlay */

    COLORREF        color_bkg;
    COLORREF        color_bkgtxt;

    LPDIRECTDRAW2        ddobject;                    /* DirectDraw object */
    LPDIRECTDRAWSURFACE2 display;                        /* Display device */
    LPDIRECTDRAWCLIPPER  clipper;             /* clipper used for blitting */
    HINSTANCE            hddraw_dll;       /* handle of the opened ddraw dll */

    picture_sys_t        *picsys;

    /* It protects the following variables */
    vlc_mutex_t    lock;
    bool           ch_wallpaper;
    bool           wallpaper_requested;
};

static picture_pool_t *Pool  (vout_display_t *, unsigned);
static void           Display(vout_display_t *, picture_t *);
static int            Control(vout_display_t *, int, va_list);
static void           Manage (vout_display_t *);

/* */
static int WallpaperCallback(vlc_object_t *, char const *,
                             vlc_value_t, vlc_value_t, void *);

static int  DirectXOpen(vout_display_t *, video_format_t *fmt);
static void DirectXClose(vout_display_t *);

static int  DirectXLock(picture_t *);
static void DirectXUnlock(picture_t *);

static int DirectXUpdateOverlay(vout_display_t *, LPDIRECTDRAWSURFACE2 surface);

static void WallpaperChange(vout_display_t *vd, bool use_wallpaper);

/** This function allocates and initialize the DirectX vout display.
 */
static int Open(vout_display_t *vd, const vout_display_cfg_t *cfg,
                video_format_t *fmtp, vlc_video_context *context)
{
    vout_display_sys_t *sys;

    /* Allocate structure */
    vd->sys = sys = calloc(1, sizeof(*sys));
    if (!sys)
        return VLC_ENOMEM;

    /* Load direct draw DLL */
    sys->hddraw_dll = LoadLibrary(_T("DDRAW.DLL"));
    if (!sys->hddraw_dll) {
        msg_Warn(vd, "DirectXInitDDraw failed loading ddraw.dll");
        free(sys);
        return VLC_EGENERIC;
    }

    /* */
    sys->use_wallpaper = var_CreateGetBool(vd, "video-wallpaper");
    /* FIXME */
    sys->sys.use_overlay = false;//var_CreateGetBool(vd, "overlay"); /* FIXME */
    sys->restore_overlay = false;
    var_Create(vd, "directx-device", VLC_VAR_STRING | VLC_VAR_DOINHERIT);
