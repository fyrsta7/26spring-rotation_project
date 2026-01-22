    return CommonControl(vd, query, args);
}

static const struct vout_window_operations embedVideoWindow_Ops =
{
};

static vout_window_t *EmbedVideoWindow_Create(vout_display_t *vd)
{
    vout_display_sys_t *sys = vd->sys;

    if (!sys->sys.hvideownd)
        return NULL;

    vout_window_t *wnd = vlc_object_create(vd, sizeof(vout_window_t));
    if (!wnd)
        return NULL;

    wnd->type = VOUT_WINDOW_TYPE_HWND;
    wnd->handle.hwnd = sys->sys.hvideownd;
    wnd->ops = &embedVideoWindow_Ops;
    return wnd;
}

/**
 * It creates an OpenGL vout display.
 */
static int Open(vout_display_t *vd, const vout_display_cfg_t *cfg,
                video_format_t *fmtp, vlc_video_context *context)
{
    vout_display_sys_t *sys;

    /* do not use OpenGL on XP unless forced */
    if(!vd->obj.force && !IsWindowsVistaOrGreater())
        return VLC_EGENERIC;

    /* Allocate structure */
    vd->sys = sys = calloc(1, sizeof(*sys));
    if (!sys)
        return VLC_ENOMEM;

    /* */
    if (CommonInit(vd, false, cfg))
        goto error;

    if (!sys->sys.b_windowless)
        EventThreadUpdateTitle(sys->sys.event, VOUT_TITLE " (OpenGL output)");

    vout_window_t *surface = EmbedVideoWindow_Create(vd);
    if (!surface)
        goto error;

    char *modlist = var_InheritString(surface, "gl");
    sys->gl = vlc_gl_Create (surface, VLC_OPENGL, modlist);
    free(modlist);
    if (!sys->gl)
    {
        vlc_object_release(surface);
        goto error;
    }

    vlc_gl_Resize (sys->gl, cfg->display.width, cfg->display.height);

    video_format_t fmt = *fmtp;
    const vlc_fourcc_t *subpicture_chromas;
