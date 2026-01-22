    vc->vid_width    = width;
    vc->vid_height   = height;

    vc->rgb_mode = get_rgb_format(format) >= 0;

    vc->deint = vc->rgb_mode ? 0 : vc->user_deint;

    free_video_specific(vo);

    vo_x11_config_vo_window(vo, NULL, vo->dx, vo->dy, d_width, d_height,
                            flags, "vdpau");

    if (initialize_vdpau_objects(vo) < 0)
        return -1;

    return 0;
}

static struct bitmap_packer *make_packer(struct vo *vo, VdpRGBAFormat format)
{
    struct vdpctx *vc = vo->priv;
    struct vdp_functions *vdp = vc->vdp;

    struct bitmap_packer *packer = talloc_zero(vo, struct bitmap_packer);
    uint32_t w_max = 0, h_max = 0;
    VdpStatus vdp_st = vdp->
        bitmap_surface_query_capabilities(vc->vdp_device, format,
                                          &(VdpBool){0}, &w_max, &h_max);
    CHECK_ST_WARNING("Query to get max OSD surface size failed");
    packer->w_max = w_max;
    packer->h_max = h_max;
    return packer;
}

static void draw_osd_part(struct vo *vo, int index)
{
    struct vdpctx *vc = vo->priv;
    struct vdp_functions *vdp = vc->vdp;
    VdpStatus vdp_st;
    struct osd_bitmap_surface *sfc = &vc->osd_surfaces[index];
    VdpOutputSurface output_surface = vc->output_surfaces[vc->surface_num];
    int i;
