
static void window_move(struct vo_wayland_state *wl, uint32_t serial)
{
    if (wl->display.shell)
        wl_shell_surface_move(wl->window.shell_surface, wl->input.seat, serial);
}

static void window_set_toplevel(struct vo_wayland_state *wl)
{
    if (wl->display.shell)
        wl_shell_surface_set_toplevel(wl->window.shell_surface);
}

static void window_set_title(struct vo_wayland_state *wl, const char *title)
{
    if (wl->display.shell)
        wl_shell_surface_set_title(wl->window.shell_surface, title);
}

static void schedule_resize(struct vo_wayland_state *wl,
                            uint32_t edges,
                            int32_t width,
                            int32_t height)
{
    int32_t minimum_size = 150;
    int32_t x, y;
    float temp_aspect = width / (float) MPMAX(height, 1);
    float win_aspect = wl->window.aspect;
    if (win_aspect <= 0)
        win_aspect = 1;

    MP_DBG(wl, "schedule resize: %dx%d\n", width, height);

    if (width < minimum_size)
        width = minimum_size;

    if (height < minimum_size)
        height = minimum_size;

    // don't keep the aspect ration in fullscreen mode, because the compositor
    // shows the desktop in the border regions if the video has not the same
    // aspect ration as the screen
    /* if only the height is changed we have to calculate the width
     * in any other case we calculate the height */
    switch (edges) {
        case WL_SHELL_SURFACE_RESIZE_TOP:
        case WL_SHELL_SURFACE_RESIZE_BOTTOM:
            width = win_aspect * height;
            break;
        case WL_SHELL_SURFACE_RESIZE_LEFT:
        case WL_SHELL_SURFACE_RESIZE_RIGHT:
        case WL_SHELL_SURFACE_RESIZE_TOP_LEFT:    // just a preference
        case WL_SHELL_SURFACE_RESIZE_TOP_RIGHT:
        case WL_SHELL_SURFACE_RESIZE_BOTTOM_LEFT:
        case WL_SHELL_SURFACE_RESIZE_BOTTOM_RIGHT:
            height = (1 / win_aspect) * width;
            break;
        default:
            if (wl->window.aspect < temp_aspect)
                width = wl->window.aspect * height;
            else
                height = (1 / win_aspect) * width;
            break;
    }

    if (edges & WL_SHELL_SURFACE_RESIZE_LEFT)
        x = wl->window.width - width;
    else
        x = 0;

    if (edges & WL_SHELL_SURFACE_RESIZE_TOP)
        y = wl->window.height - height;
