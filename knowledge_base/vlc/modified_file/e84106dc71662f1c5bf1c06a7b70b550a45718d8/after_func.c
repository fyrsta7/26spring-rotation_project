        *width = cfg->window_props.width;
        *height = cfg->window_props.height;
        /* The dimensions are not initialized from the source format */
        from_source = false;
    }
    /* Use the original video size */
    else if (source->i_sar_num >= source->i_sar_den) {
        *width  = (int64_t)source->i_visible_width * source->i_sar_num * cfg->display.sar.den / source->i_sar_den / cfg->display.sar.num;
        *height = source->i_visible_height;
    } else {
        *width  = source->i_visible_width;
        *height = (int64_t)source->i_visible_height * source->i_sar_den * cfg->display.sar.num / source->i_sar_num / cfg->display.sar.den;
    }

    *width  = *width  * cfg->zoom.num / cfg->zoom.den;
    *height = *height * cfg->zoom.num / cfg->zoom.den;

    if (from_source && ORIENT_IS_SWAP(source->orientation)) {
        /* Apply the source orientation only if the dimensions are initialized
         * from the source format */
        unsigned store = *width;
        *width = *height;
        *height = store;
    }
}

/* */
void vout_display_PlacePicture(vout_display_place_t *place,
                               const video_format_t *source,
                               const vout_display_cfg_t *cfg)
{
    /* vout_display_PlacePicture() is called from vd plugins. They should not
     * care about the initial window properties. */
    assert(cfg->window_props.width == 0 && cfg->window_props.height == 0);

    /* */
    memset(place, 0, sizeof(*place));
    if (cfg->display.width == 0 || cfg->display.height == 0)
        return;

    /* */
    unsigned display_width;
    unsigned display_height;

    video_format_t source_rot;
    video_format_ApplyRotation(&source_rot, source);
    source = &source_rot;

    if (cfg->is_display_filled) {
        display_width  = cfg->display.width;
        display_height = cfg->display.height;
    } else
        vout_display_GetDefaultDisplaySize(&display_width, &display_height,
                                           source, cfg);

    const unsigned width  = source->i_visible_width;
    const unsigned height = source->i_visible_height;
    /* Compute the height if we use the width to fill up display_width */
    const int64_t scaled_height = (int64_t)height * display_width  * cfg->display.sar.num * source->i_sar_den / (width  * source->i_sar_num * cfg->display.sar.den);
    /* And the same but switching width/height */
    const int64_t scaled_width  = (int64_t)width  * display_height * cfg->display.sar.den * source->i_sar_num / (height * source->i_sar_den * cfg->display.sar.num);

    if (source->projection_mode == PROJECTION_MODE_RECTANGULAR) {
        /* We keep the solution that avoid filling outside the display */
        if (scaled_width <= cfg->display.width) {
            place->width  = scaled_width;
            place->height = display_height;
        } else {
            place->width  = display_width;
            place->height = scaled_height;
        }
    } else {
        /* No need to preserve an aspect ratio for 360 video.
         * They can fill the display. */
