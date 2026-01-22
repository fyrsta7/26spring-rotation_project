{
    if (!p->original_image)
        return false;

    draw_image(p->vo, p->original_image);
    return true;
}

static mp_image_t *get_screenshot(struct priv *p)
{
    if (!p->original_image)
        return NULL;

    return mp_image_new_ref(p->original_image);
}

static bool resize(struct priv *p)
{
    struct vo_wayland_state *wl = p->wl;
    int32_t x = wl->window.sh_x;
    int32_t y = wl->window.sh_y;
    wl->vo->dwidth = wl->window.sh_width;
    wl->vo->dheight = wl->window.sh_height;

    vo_get_src_dst_rects(p->vo, &p->src, &p->dst, &p->osd);
    p->src_w = p->src.x1 - p->src.x0;
    p->src_h = p->src.y1 - p->src.y0;
    p->dst_w = p->dst.x1 - p->dst.x0;
    p->dst_h = p->dst.y1 - p->dst.y0;

    MP_DBG(p->vo, "resizing %dx%d -> %dx%d\n", wl->window.width,
                                               wl->window.height,
                                               p->dst_w,
                                               p->dst_h);

    if (x != 0)
        x = wl->window.width - p->dst_w;

    if (y != 0)
        y = wl->window.height - p->dst_h;

    mp_sws_set_from_cmdline(p->sws);
    p->sws->src = p->in_format;
    p->sws->dst = (struct mp_image_params) {
        .imgfmt = p->pref_format->mp_fmt,
        .w = p->dst_w,
        .h = p->dst_h,
        .d_w = p->dst_w,
        .d_h = p->dst_h,
    };

    mp_image_params_guess_csp(&p->sws->dst);

    if (mp_sws_reinit(p->sws) < 0)
        return false;

    if (!reinit_shm_buffers(p, p->dst_w, p->dst_h)) {
        MP_ERR(p->vo, "failed to resize buffers\n");
        return false;
    }

    wl->window.width = p->dst_w;
    wl->window.height = p->dst_h;

    // if no alpha enabled format is used then create an opaque region to allow
    // the compositor to optimize the drawing of the window
    if (!p->enable_alpha) {
