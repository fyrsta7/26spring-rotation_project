            // the textures are bound in-order and starting at 0, we just
            // assert to make sure this is the case (which it should always be)
            int id = pass_bind(p, img);
            assert(id == i);
        }

        MP_TRACE(p, "inter frame dur: %f vsync: %f, mix: %f\n",
                 t->ideal_frame_duration, t->vsync_interval, mix);
        p->is_interpolated = true;
    }
    pass_draw_to_screen(p, fbo);

    p->frames_drawn += 1;
}

void gl_video_render_frame(struct gl_video *p, struct vo_frame *frame,
                           struct ra_fbo fbo, int flags)
{
    gl_video_update_options(p);

    struct mp_rect target_rc = {0, 0, fbo.tex->params.w, fbo.tex->params.h};

    p->broken_frame = false;

    bool has_frame = !!frame->current;

    if (!has_frame || !mp_rect_equals(&p->dst_rect, &target_rc)) {
        struct m_color c = p->clear_color;
        float clear_color[4] = {c.r / 255.0, c.g / 255.0, c.b / 255.0, c.a / 255.0};
        p->ra->fns->clear(p->ra, fbo.tex, clear_color, &target_rc);
    }

    if (p->hwdec_overlay) {
        if (has_frame) {
            float *color = p->hwdec_overlay->overlay_colorkey;
            p->ra->fns->clear(p->ra, fbo.tex, color, &p->dst_rect);
        }

        p->hwdec_overlay->driver->overlay_frame(p->hwdec_overlay, frame->current,
                                                &p->src_rect, &p->dst_rect,
                                                frame->frame_id != p->image.id);

        if (frame->current)
            p->osd_pts = frame->current->pts;

        // Disable GL rendering
        has_frame = false;
    }

    if (has_frame) {
        bool interpolate = p->opts.interpolation && frame->display_synced &&
                           (p->frames_drawn || !frame->still);
        if (interpolate) {
            double ratio = frame->ideal_frame_duration / frame->vsync_interval;
            if (fabs(ratio - 1.0) < p->opts.interpolation_threshold)
                interpolate = false;
        }

        if (interpolate) {
            gl_video_interpolate_frame(p, frame, fbo, flags);
        } else {
            bool is_new = frame->frame_id != p->image.id;

            // Redrawing a frame might update subtitles.
            if (frame->still && p->opts.blend_subs)
                is_new = true;

            if (is_new || !p->output_tex_valid) {
                p->output_tex_valid = false;

                pass_info_reset(p, !is_new);
                if (!pass_render_frame(p, frame->current, frame->frame_id, flags))
                    goto done;

                // For the non-interpolation case, we draw to a single "cache"
                // texture to speed up subsequent re-draws (if any exist)
                struct ra_fbo dest_fbo = fbo;
                bool repeats = frame->num_vsyncs > 1 && frame->display_synced;
                if ((repeats || frame->still) && !p->dumb_mode &&
                    (p->ra->caps & RA_CAP_BLIT) && fbo.tex->params.blit_dst)
                {
                    // Attempt to use the same format as the destination FBO
                    // if possible. Some RAs use a wrapped dummy format here,
                    // so fall back to the fbo_format in that case.
                    const struct ra_format *fmt = fbo.tex->params.format;
                    if (fmt->dummy_format)
                        fmt = p->fbo_format;

                    bool r = ra_tex_resize(p->ra, p->log, &p->output_tex,
                                           fbo.tex->params.w, fbo.tex->params.h,
                                           fmt);
                    if (r) {
                        dest_fbo = (struct ra_fbo) { p->output_tex };
                        p->output_tex_valid = true;
                    }
                }
                pass_draw_to_screen(p, dest_fbo);
            }

            // "output tex valid" and "output tex needed" are equivalent
            if (p->output_tex_valid && fbo.tex->params.blit_dst) {
                pass_info_reset(p, true);
                pass_describe(p, "redraw cached frame");
                struct mp_rect src = p->dst_rect;
                struct mp_rect dst = src;
                if (fbo.flip) {
                    dst.y0 = fbo.tex->params.h - src.y0;
                    dst.y1 = fbo.tex->params.h - src.y1;
                }
                timer_pool_start(p->blit_timer);
                p->ra->fns->blit(p->ra, fbo.tex, p->output_tex, &dst, &src);
                timer_pool_stop(p->blit_timer);
                pass_record(p, timer_pool_measure(p->blit_timer));
            }
        }
    }

done:

    debug_check_gl(p, "after video rendering");

    if (p->osd && (flags & (RENDER_FRAME_SUBS | RENDER_FRAME_OSD))) {
        // If we haven't actually drawn anything so far, then we technically
        // need to consider this the start of a new pass. Let's call it a
        // redraw just because, since it's basically a blank frame anyway
        if (!has_frame)
            pass_info_reset(p, true);

        int osd_flags = p->opts.blend_subs ? OSD_DRAW_OSD_ONLY : 0;
        if (!(flags & RENDER_FRAME_SUBS))
            osd_flags |= OSD_DRAW_OSD_ONLY;
        if (!(flags & RENDER_FRAME_OSD))
            osd_flags |= OSD_DRAW_SUB_ONLY;

