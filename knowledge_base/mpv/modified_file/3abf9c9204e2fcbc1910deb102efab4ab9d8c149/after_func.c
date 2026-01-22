            pass_delinearize(p->sc, p->image_params.gamma);
            p->use_linear = false;
        }
        finish_pass_fbo(p, &p->blend_subs_fbo, p->texture_w, p->texture_h,
                        FBOTEX_FUZZY);
        pass_draw_osd(p, OSD_DRAW_SUB_ONLY, vpts, rect,
                      p->texture_w, p->texture_h, p->blend_subs_fbo.fbo, false);
        pass_read_fbo(p, &p->blend_subs_fbo);
    }

    pass_opt_hook_point(p, "SCALED", NULL);

    gl_timer_stop(p->render_timer);
}

static void pass_draw_to_screen(struct gl_video *p, int fbo)
{
    gl_timer_start(p->present_timer);

    if (p->dumb_mode)
        pass_render_frame_dumb(p, fbo);

    // Adjust the overall gamma before drawing to screen
    if (p->user_gamma != 1) {
        gl_sc_uniform_f(p->sc, "user_gamma", p->user_gamma);
        GLSL(color.rgb = clamp(color.rgb, 0.0, 1.0);)
        GLSL(color.rgb = pow(color.rgb, vec3(user_gamma));)
    }

    pass_colormanage(p, p->image_params.peak, p->image_params.primaries,
                     p->use_linear ? MP_CSP_TRC_LINEAR : p->image_params.gamma);

    // Draw checkerboard pattern to indicate transparency
    if (p->has_alpha && p->opts.alpha_mode == ALPHA_BLEND_TILES) {
        GLSLF("// transparency checkerboard\n");
        GLSL(bvec2 tile = lessThan(fract(gl_FragCoord.xy / 32.0), vec2(0.5));)
        GLSL(vec3 background = vec3(tile.x == tile.y ? 1.0 : 0.75);)
        GLSL(color.rgb = mix(background, color.rgb, color.a);)
    }

    pass_opt_hook_point(p, "OUTPUT", NULL);

    pass_dither(p);
    finish_pass_direct(p, fbo, p->vp_w, p->vp_h, &p->dst_rect);

    gl_timer_stop(p->present_timer);
}

// Draws an interpolate frame to fbo, based on the frame timing in t
static void gl_video_interpolate_frame(struct gl_video *p, struct vo_frame *t,
                                       int fbo)
{
    int vp_w = p->dst_rect.x1 - p->dst_rect.x0,
        vp_h = p->dst_rect.y1 - p->dst_rect.y0;

    // Reset the queue completely if this is a still image, to avoid any
    // interpolation artifacts from surrounding frames when unpausing or
    // framestepping
    if (t->still)
        gl_video_reset_surfaces(p);

    // First of all, figure out if we have a frame available at all, and draw
    // it manually + reset the queue if not
    if (p->surfaces[p->surface_now].pts == MP_NOPTS_VALUE) {
        if (!gl_video_upload_image(p, t->current))
            return;
        pass_render_frame(p);
        finish_pass_fbo(p, &p->surfaces[p->surface_now].fbotex,
                        vp_w, vp_h, FBOTEX_FUZZY);
        p->surfaces[p->surface_now].pts = p->image.mpi->pts;
        p->surface_idx = p->surface_now;
    }

    // Find the right frame for this instant
    if (t->current && t->current->pts != MP_NOPTS_VALUE) {
        int next = fbosurface_wrap(p->surface_now + 1);
        while (p->surfaces[next].pts != MP_NOPTS_VALUE &&
               p->surfaces[next].pts > p->surfaces[p->surface_now].pts &&
               p->surfaces[p->surface_now].pts < t->current->pts)
        {
            p->surface_now = next;
            next = fbosurface_wrap(next + 1);
        }
    }

    // Figure out the queue size. For illustration, a filter radius of 2 would
    // look like this: _ A [B] C D _
    // A is surface_bse, B is surface_now, C is surface_now+1 and D is
    // surface_end.
    struct scaler *tscale = &p->scaler[SCALER_TSCALE];
    reinit_scaler(p, tscale, &p->opts.scaler[SCALER_TSCALE], 1, tscale_sizes);
    bool oversample = strcmp(tscale->conf.kernel.name, "oversample") == 0;
    int size;

    if (oversample) {
        size = 2;
    } else {
        assert(tscale->kernel && !tscale->kernel->polar);
        size = ceil(tscale->kernel->size);
        assert(size <= TEXUNIT_VIDEO_NUM);
    }

    int radius = size/2;
    int surface_now = p->surface_now;
    int surface_bse = fbosurface_wrap(surface_now - (radius-1));
    int surface_end = fbosurface_wrap(surface_now + radius);
    assert(fbosurface_wrap(surface_bse + size-1) == surface_end);

    // Render new frames while there's room in the queue. Note that technically,
    // this should be done before the step where we find the right frame, but
    // it only barely matters at the very beginning of playback, and this way
    // makes the code much more linear.
    int surface_dst = fbosurface_wrap(p->surface_idx + 1);
    for (int i = 0; i < t->num_frames; i++) {
        // Avoid overwriting data we might still need
        if (surface_dst == surface_bse - 1)
            break;

        struct mp_image *f = t->frames[i];
        if (!mp_image_params_equal(&f->params, &p->real_image_params) ||
            f->pts == MP_NOPTS_VALUE)
            continue;

        if (f->pts > p->surfaces[p->surface_idx].pts) {
            if (!gl_video_upload_image(p, f))
                return;
            pass_render_frame(p);
            finish_pass_fbo(p, &p->surfaces[surface_dst].fbotex,
                            vp_w, vp_h, FBOTEX_FUZZY);
            p->surfaces[surface_dst].pts = f->pts;
            p->surface_idx = surface_dst;
            surface_dst = fbosurface_wrap(surface_dst + 1);
        }
    }

    // Figure out whether the queue is "valid". A queue is invalid if the
    // frames' PTS is not monotonically increasing. Anything else is invalid,
    // so avoid blending incorrect data and just draw the latest frame as-is.
    // Possible causes for failure of this condition include seeks, pausing,
    // end of playback or start of playback.
    bool valid = true;
    for (int i = surface_bse, ii; valid && i != surface_end; i = ii) {
        ii = fbosurface_wrap(i + 1);
        if (p->surfaces[i].pts == MP_NOPTS_VALUE ||
            p->surfaces[ii].pts == MP_NOPTS_VALUE)
        {
            valid = false;
        } else if (p->surfaces[ii].pts < p->surfaces[i].pts) {
            valid = false;
            MP_DBG(p, "interpolation queue underrun\n");
        }
    }

    // Update OSD PTS to synchronize subtitles with the displayed frame
    p->osd_pts = p->surfaces[surface_now].pts;

    // Finally, draw the right mix of frames to the screen.
    if (!valid || t->still) {
        // surface_now is guaranteed to be valid, so we can safely use it.
        pass_read_fbo(p, &p->surfaces[surface_now].fbotex);
        p->is_interpolated = false;
    } else {
        double mix = t->vsync_offset / t->ideal_frame_duration;
        // The scaler code always wants the fcoord to be between 0 and 1,
        // so we try to adjust by using the previous set of N frames instead
        // (which requires some extra checking to make sure it's valid)
        if (mix < 0.0) {
            int prev = fbosurface_wrap(surface_bse - 1);
            if (p->surfaces[prev].pts != MP_NOPTS_VALUE &&
                p->surfaces[prev].pts < p->surfaces[surface_bse].pts)
            {
                mix += 1.0;
                surface_bse = prev;
            } else {
                mix = 0.0; // at least don't blow up, this should only
                           // ever happen at the start of playback
            }
        }

        // Blend the frames together
        if (oversample) {
            double vsync_dist = t->vsync_interval / t->ideal_frame_duration,
                   threshold = tscale->conf.kernel.params[0];
            threshold = isnan(threshold) ? 0.0 : threshold;
            mix = (1 - mix) / vsync_dist;
            mix = mix <= 0 + threshold ? 0 : mix;
            mix = mix >= 1 - threshold ? 1 : mix;
            mix = 1 - mix;
            gl_sc_uniform_f(p->sc, "inter_coeff", mix);
            GLSL(color = mix(texture(texture0, texcoord0),
                             texture(texture1, texcoord1),
                             inter_coeff);)
        } else {
            gl_sc_uniform_f(p->sc, "fcoord", mix);
            pass_sample_separated_gen(p->sc, tscale, 0, 0);
        }

        // Load all the required frames
        for (int i = 0; i < size; i++) {
            struct img_tex img =
                img_tex_fbo(&p->surfaces[fbosurface_wrap(surface_bse+i)].fbotex,
                            PLANE_RGB, p->components);
            // Since the code in pass_sample_separated currently assumes
            // the textures are bound in-order and starting at 0, we just
            // assert to make sure this is the case (which it should always be)
            int id = pass_bind(p, img);
            assert(id == i);
        }

        MP_DBG(p, "inter frame dur: %f vsync: %f, mix: %f\n",
               t->ideal_frame_duration, t->vsync_interval, mix);
        p->is_interpolated = true;
    }
    pass_draw_to_screen(p, fbo);

    p->frames_drawn += 1;
}

static void timer_dbg(struct gl_video *p, const char *name, struct gl_timer *t)
{
    if (gl_timer_sample_count(t) > 0) {
        MP_DBG(p, "%s time: last %dus avg %dus peak %dus\n", name,
               (int)gl_timer_last_us(t),
               (int)gl_timer_avg_us(t),
               (int)gl_timer_peak_us(t));
    }
}

// (fbo==0 makes BindFramebuffer select the screen backbuffer)
void gl_video_render_frame(struct gl_video *p, struct vo_frame *frame, int fbo)
{
    GL *gl = p->gl;
    struct video_image *vimg = &p->image;

    p->broken_frame = false;

    gl->BindFramebuffer(GL_FRAMEBUFFER, fbo);

    bool has_frame = frame->current || vimg->mpi;

    if (!has_frame || p->dst_rect.x0 > 0 || p->dst_rect.y0 > 0 ||
        p->dst_rect.x1 < p->vp_w || p->dst_rect.y1 < abs(p->vp_h))
    {
        struct m_color c = p->opts.background;
        gl->ClearColor(c.r / 255.0, c.g / 255.0, c.b / 255.0, c.a / 255.0);
        gl->Clear(GL_COLOR_BUFFER_BIT);
    }

    if (has_frame) {
        gl_sc_set_vao(p->sc, &p->vao);

        bool interpolate = p->opts.interpolation && frame->display_synced &&
                           (p->frames_drawn || !frame->still);
        if (interpolate) {
            double ratio = frame->ideal_frame_duration / frame->vsync_interval;
            if (fabs(ratio - 1.0) < p->opts.interpolation_threshold)
                interpolate = false;
        }

        if (interpolate) {
            gl_video_interpolate_frame(p, frame, fbo);
        } else {
            bool is_new = !frame->redraw && !frame->repeat;
            if (is_new || !p->output_fbo_valid) {
                p->output_fbo_valid = false;

                if (!gl_video_upload_image(p, frame->current))
                    goto done;
                pass_render_frame(p);

                // For the non-interpolation case, we draw to a single "cache"
                // FBO to speed up subsequent re-draws (if any exist)
                int dest_fbo = fbo;
                if (frame->num_vsyncs > 1 && frame->display_synced &&
                    !p->dumb_mode && gl->BlitFramebuffer)
                {
                    fbotex_change(&p->output_fbo, p->gl, p->log,
                                  p->vp_w, abs(p->vp_h),
                                  p->opts.fbo_format, FBOTEX_FUZZY);
                    dest_fbo = p->output_fbo.fbo;
                    p->output_fbo_valid = true;
                }
                pass_draw_to_screen(p, dest_fbo);
            }

            // "output fbo valid" and "output fbo needed" are equivalent
            if (p->output_fbo_valid) {
                gl->BindFramebuffer(GL_READ_FRAMEBUFFER, p->output_fbo.fbo);
                gl->BindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
                struct mp_rect rc = p->dst_rect;
                if (p->vp_h < 0) {
                    rc.y1 = -p->vp_h - p->dst_rect.y0;
                    rc.y0 = -p->vp_h - p->dst_rect.y1;
                }
                gl->BlitFramebuffer(rc.x0, rc.y0, rc.x1, rc.y1,
                                    rc.x0, rc.y0, rc.x1, rc.y1,
                                    GL_COLOR_BUFFER_BIT, GL_NEAREST);
                gl->BindFramebuffer(GL_READ_FRAMEBUFFER, 0);
                gl->BindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
            }
        }
    }

done:

    unmap_current_image(p);

    debug_check_gl(p, "after video rendering");

    gl->BindFramebuffer(GL_FRAMEBUFFER, fbo);

    if (p->osd) {
        pass_draw_osd(p, p->opts.blend_subs ? OSD_DRAW_OSD_ONLY : 0,
                      p->osd_pts, p->osd_rect, p->vp_w, p->vp_h, fbo, true);
        debug_check_gl(p, "after OSD rendering");
    }
    gl->UseProgram(0);

    if (gl_sc_error_state(p->sc) || p->broken_frame) {
        // Make the screen solid blue to make it visually clear that an
        // error has occurred
        gl->ClearColor(0.0, 0.05, 0.5, 1.0);
        gl->Clear(GL_COLOR_BUFFER_BIT);
    }

    gl->BindFramebuffer(GL_FRAMEBUFFER, 0);

    // The playloop calls this last before waiting some time until it decides
    // to call flip_page(). Tell OpenGL to start execution of the GPU commands
    // while we sleep (this happens asynchronously).
    gl->Flush();

    p->frames_rendered++;

    // Report performance metrics
    timer_dbg(p, "upload", p->upload_timer);
    timer_dbg(p, "render", p->render_timer);
    timer_dbg(p, "present", p->present_timer);
}

// vp_w/vp_h is the implicit size of the target framebuffer.
// vp_h can be negative to flip the screen.
void gl_video_resize(struct gl_video *p, int vp_w, int vp_h,
                     struct mp_rect *src, struct mp_rect *dst,
                     struct mp_osd_res *osd)
{
    p->src_rect = *src;
    p->dst_rect = *dst;
    p->osd_rect = *osd;
    p->vp_w = vp_w;
    p->vp_h = vp_h;

    gl_video_reset_surfaces(p);
    gl_video_setup_hooks(p);

    if (p->osd)
        mpgl_osd_resize(p->osd, p->osd_rect, p->image_params.stereo_out);
}

static struct voctrl_performance_entry gl_video_perfentry(struct gl_timer *t)
{
    return (struct voctrl_performance_entry) {
        .last = gl_timer_last_us(t),
        .avg  = gl_timer_avg_us(t),
        .peak = gl_timer_peak_us(t),
    };
}

struct voctrl_performance_data gl_video_perfdata(struct gl_video *p)
{
    return (struct voctrl_performance_data) {
        .upload = gl_video_perfentry(p->upload_timer),
        .render = gl_video_perfentry(p->render_timer),
        .present = gl_video_perfentry(p->present_timer),
    };
}

// This assumes nv12, with textures set to GL_NEAREST filtering.
static void reinterleave_vdpau(struct gl_video *p, struct gl_hwdec_frame *frame)
{
    struct gl_hwdec_frame res = {0};
    for (int n = 0; n < 2; n++) {
        struct fbotex *fbo = &p->vdpau_deinterleave_fbo[n];
        // This is an array of the 2 to-merge planes.
        struct gl_hwdec_plane *src = &frame->planes[n * 2];
        int w = src[0].tex_w;
        int h = src[0].tex_h;
        int ids[2];
        for (int t = 0; t < 2; t++) {
            ids[t] = pass_bind(p, (struct img_tex){
                .gl_tex = src[t].gl_texture,
                .gl_target = src[t].gl_target,
                .multiplier = 1.0,
                .transform = identity_trans,
                .tex_w = w,
                .tex_h = h,
                .w = w,
                .h = h,
            });
        }

        GLSLF("color = fract(gl_FragCoord.y / 2) < 0.5\n");
        GLSLF("      ? texture(texture%d, texcoord%d)\n", ids[0], ids[0]);
        GLSLF("      : texture(texture%d, texcoord%d);", ids[1], ids[1]);

        fbotex_change(fbo, p->gl, p->log, w, h * 2, n == 0 ? GL_R8 : GL_RG8, 0);

        finish_pass_direct(p, fbo->fbo, fbo->rw, fbo->rh,
                           &(struct mp_rect){0, 0, w, h * 2});

        res.planes[n] = (struct gl_hwdec_plane){
            .gl_texture = fbo->texture,
            .gl_target = GL_TEXTURE_2D,
            .tex_w = w,
            .tex_h = h * 2,
        };
    }
    *frame = res;
}

// Returns false on failure.
static bool gl_video_upload_image(struct gl_video *p, struct mp_image *mpi)
{
    GL *gl = p->gl;
    struct video_image *vimg = &p->image;

    unref_current_image(p);

    mpi = mp_image_new_ref(mpi);
    if (!mpi)
        goto error;

    vimg->mpi = mpi;
    p->osd_pts = mpi->pts;
    p->frames_uploaded++;

    if (p->hwdec_active) {
        // Hardware decoding
        struct gl_hwdec_frame gl_frame = {0};
        gl_timer_start(p->upload_timer);
        bool ok = p->hwdec->driver->map_frame(p->hwdec, vimg->mpi, &gl_frame) >= 0;
        gl_timer_stop(p->upload_timer);
        vimg->hwdec_mapped = true;
        if (ok) {
            struct mp_image layout = {0};
            mp_image_set_params(&layout, &p->image_params);
            if (gl_frame.vdpau_fields)
                reinterleave_vdpau(p, &gl_frame);
            for (int n = 0; n < p->plane_count; n++) {
                struct gl_hwdec_plane *plane = &gl_frame.planes[n];
                vimg->planes[n] = (struct texplane){
                    .w = mp_image_plane_w(&layout, n),
                    .h = mp_image_plane_h(&layout, n),
                    .tex_w = plane->tex_w,
                    .tex_h = plane->tex_h,
                    .gl_target = plane->gl_target,
                    .gl_texture = plane->gl_texture,
                };
                snprintf(vimg->planes[n].swizzle, sizeof(vimg->planes[n].swizzle),
                         "%s", plane->swizzle);
            }
        } else {
            MP_FATAL(p, "Mapping hardware decoded surface failed.\n");
            goto error;
        }
        return true;
    }

    // Software decoding
    assert(mpi->num_planes == p->plane_count);

    gl_timer_start(p->upload_timer);


    for (int n = 0; n < p->plane_count; n++) {
        struct texplane *plane = &vimg->planes[n];

        plane->flipped = mpi->stride[0] < 0;

        gl->BindTexture(plane->gl_target, plane->gl_texture);
        gl_pbo_upload_tex(&plane->pbo, gl, p->opts.pbo, plane->gl_target,
                          plane->gl_format, plane->gl_type, plane->w, plane->h,
                          mpi->planes[n], mpi->stride[n],
                          0, 0, plane->w, plane->h);
        gl->BindTexture(plane->gl_target, 0);
    }

    gl_timer_stop(p->upload_timer);

    return true;

error:
    unref_current_image(p);
    p->broken_frame = true;
    return false;
}

static bool test_fbo(struct gl_video *p, GLint format)
{
    GL *gl = p->gl;
    bool success = false;
    MP_VERBOSE(p, "Testing FBO format 0x%x\n", (unsigned)format);
    struct fbotex fbo = {0};
    if (fbotex_init(&fbo, p->gl, p->log, 16, 16, format)) {
        gl->BindFramebuffer(GL_FRAMEBUFFER, fbo.fbo);
        gl->BindFramebuffer(GL_FRAMEBUFFER, 0);
        success = true;
    }
    fbotex_uninit(&fbo);
    gl_check_error(gl, p->log, "FBO test");
    return success;
}

// Return whether dumb-mode can be used without disabling any features.
// Essentially, vo_opengl with mostly default settings will return true.
static bool check_dumb_mode(struct gl_video *p)
{
    struct gl_video_opts *o = &p->opts;
    if (p->use_integer_conversion)
        return false;
    if (o->dumb_mode)
        return true;
    if (o->target_prim || o->target_trc || o->linear_scaling ||
        o->correct_downscaling || o->sigmoid_upscaling || o->interpolation ||
        o->blend_subs || o->deband || o->unsharp)
        return false;
    // check remaining scalers (tscale is already implicitly excluded above)
    for (int i = 0; i < SCALER_COUNT; i++) {
        if (i != SCALER_TSCALE) {
            const char *name = o->scaler[i].kernel.name;
            if (name && strcmp(name, "bilinear") != 0)
                return false;
        }
    }
    if (o->pre_shaders && o->pre_shaders[0])
        return false;
    if (o->post_shaders && o->post_shaders[0])
        return false;
    if (o->user_shaders && o->user_shaders[0])
        return false;
    if (p->use_lut_3d)
        return false;
    return true;
}

// Disable features that are not supported with the current OpenGL version.
static void check_gl_features(struct gl_video *p)
{
    GL *gl = p->gl;
    bool have_float_tex = !!gl_find_float16_format(gl, 1);
    bool have_3d_tex = gl->mpgl_caps & MPGL_CAP_3D_TEX;
    bool have_mglsl = gl->glsl_version >= 130; // modern GLSL (1st class arrays etc.)
    bool have_texrg = gl->mpgl_caps & MPGL_CAP_TEX_RG;
    bool have_tex16 = !gl->es || (gl->mpgl_caps & MPGL_CAP_EXT16);

    const GLint auto_fbo_fmts[] = {GL_RGBA16, GL_RGBA16F, GL_RGB10_A2,
                                   GL_RGBA8, 0};
    GLint user_fbo_fmts[] = {p->opts.fbo_format, 0};
    const GLint *fbo_fmts = user_fbo_fmts[0] ? user_fbo_fmts : auto_fbo_fmts;
    bool have_fbo = false;
    for (int n = 0; fbo_fmts[n]; n++) {
        GLint fmt = fbo_fmts[n];
        const struct gl_format *f = gl_find_internal_format(gl, fmt);
        if (f && (f->flags & F_CF) == F_CF && test_fbo(p, fmt)) {
            MP_VERBOSE(p, "Using FBO format 0x%x.\n", (unsigned)fmt);
            have_fbo = true;
            p->opts.fbo_format = fmt;
            break;
        }
    }

    if (!gl->MapBufferRange && p->opts.pbo) {
        p->opts.pbo = 0;
        MP_WARN(p, "Disabling PBOs (GL2.1/GLES2 unsupported).\n");
    }

    p->forced_dumb_mode = p->opts.dumb_mode || !have_fbo || !have_texrg;
    bool voluntarily_dumb = check_dumb_mode(p);
    if (p->forced_dumb_mode || voluntarily_dumb) {
        if (voluntarily_dumb) {
            MP_VERBOSE(p, "No advanced processing required. Enabling dumb mode.\n");
        } else if (!p->opts.dumb_mode) {
            MP_WARN(p, "High bit depth FBOs unsupported. Enabling dumb mode.\n"
                       "Most extended features will be disabled.\n");
        }
        p->dumb_mode = true;
        p->use_lut_3d = false;
        // Most things don't work, so whitelist all options that still work.
        struct gl_video_opts new_opts = {
            .gamma = p->opts.gamma,
            .gamma_auto = p->opts.gamma_auto,
            .pbo = p->opts.pbo,
            .fbo_format = p->opts.fbo_format,
            .alpha_mode = p->opts.alpha_mode,
            .use_rectangle = p->opts.use_rectangle,
            .background = p->opts.background,
            .dither_algo = DITHER_NONE,
            .target_brightness = p->opts.target_brightness,
            .hdr_tone_mapping = p->opts.hdr_tone_mapping,
            .tone_mapping_param = p->opts.tone_mapping_param,
        };
        for (int n = 0; n < SCALER_COUNT; n++)
            new_opts.scaler[n] = gl_video_opts_def.scaler[n];
        set_options(p, &new_opts);
        return;
    }
    p->dumb_mode = false;

    // Normally, we want to disable them by default if FBOs are unavailable,
    // because they will be slow (not critically slow, but still slower).
    // Without FP textures, we must always disable them.
    // I don't know if luminance alpha float textures exist, so disregard them.
    for (int n = 0; n < SCALER_COUNT; n++) {
        const struct filter_kernel *kernel =
            mp_find_filter_kernel(p->opts.scaler[n].kernel.name);
        if (kernel) {
            char *reason = NULL;
            if (!have_float_tex)
                reason = "(float tex. missing)";
            if (!have_mglsl)
                reason = "(GLSL version too old)";
            if (reason) {
                MP_WARN(p, "Disabling scaler #%d %s %s.\n", n,
                        p->opts.scaler[n].kernel.name, reason);
                // p->opts is a copy of p->opts_alloc => we can just mess with it.
                p->opts.scaler[n].kernel.name = "bilinear";
                if (n == SCALER_TSCALE)
                    p->opts.interpolation = 0;
            }
        }
    }

    // GLES3 doesn't provide filtered 16 bit integer textures
    // GLES2 doesn't even provide 3D textures
    if (p->use_lut_3d && (!have_3d_tex || !have_tex16)) {
        p->use_lut_3d = false;
        MP_WARN(p, "Disabling color management (no RGB16 3D textures).\n");
    }

    int use_cms = p->opts.target_prim != MP_CSP_PRIM_AUTO ||
                  p->opts.target_trc != MP_CSP_TRC_AUTO || p->use_lut_3d;

    // mix() is needed for some gamma functions
    if (!have_mglsl && (p->opts.linear_scaling || p->opts.sigmoid_upscaling)) {
        p->opts.linear_scaling = false;
        p->opts.sigmoid_upscaling = false;
        MP_WARN(p, "Disabling linear/sigmoid scaling (GLSL version too old).\n");
    }
    if (!have_mglsl && use_cms) {
        p->opts.target_prim = MP_CSP_PRIM_AUTO;
        p->opts.target_trc = MP_CSP_TRC_AUTO;
        p->use_lut_3d = false;
        MP_WARN(p, "Disabling color management (GLSL version too old).\n");
    }
    if (!have_mglsl && p->opts.deband) {
        p->opts.deband = 0;
        MP_WARN(p, "Disabling debanding (GLSL version too old).\n");
    }
}

static void init_gl(struct gl_video *p)
{
    GL *gl = p->gl;

    debug_check_gl(p, "before init_gl");

    gl->Disable(GL_DITHER);

    gl_vao_init(&p->vao, gl, sizeof(struct vertex), vertex_vao);

    gl_video_set_gl_state(p);

    // Test whether we can use 10 bit. Hope that testing a single format/channel
    // is good enough (instead of testing all 1-4 channels variants etc.).
    const struct gl_format *fmt = gl_find_unorm_format(gl, 2, 1);
    if (gl->GetTexLevelParameteriv && fmt) {
        GLuint tex;
        gl->GenTextures(1, &tex);
        gl->BindTexture(GL_TEXTURE_2D, tex);
        gl->TexImage2D(GL_TEXTURE_2D, 0, fmt->internal_format, 64, 64, 0,
                       fmt->format, fmt->type, NULL);
        GLenum pname = 0;
        switch (fmt->format) {
        case GL_RED:        pname = GL_TEXTURE_RED_SIZE; break;
        case GL_LUMINANCE:  pname = GL_TEXTURE_LUMINANCE_SIZE; break;
        }
        GLint param = 0;
        if (pname)
            gl->GetTexLevelParameteriv(GL_TEXTURE_2D, 0, pname, &param);
        if (param) {
            MP_VERBOSE(p, "16 bit texture depth: %d.\n", (int)param);
            p->texture_16bit_depth = param;
        }
        gl->DeleteTextures(1, &tex);
    }

    if ((gl->es >= 300 || gl->version) && (gl->mpgl_caps & MPGL_CAP_FB)) {
        gl->BindFramebuffer(GL_FRAMEBUFFER, gl->main_fb);

        GLenum obj = gl->version ? GL_BACK_LEFT : GL_BACK;
        if (gl->main_fb)
            obj = GL_COLOR_ATTACHMENT0;

        GLint depth_r = -1, depth_g = -1, depth_b = -1;

        gl->GetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, obj,
                            GL_FRAMEBUFFER_ATTACHMENT_RED_SIZE, &depth_r);
        gl->GetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, obj,
                            GL_FRAMEBUFFER_ATTACHMENT_GREEN_SIZE, &depth_g);
        gl->GetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, obj,
                            GL_FRAMEBUFFER_ATTACHMENT_BLUE_SIZE, &depth_b);

        MP_VERBOSE(p, "Reported display depth: R=%d, G=%d, B=%d\n",
                   depth_r, depth_g, depth_b);

        p->fb_depth = depth_g > 0 ? depth_g : 8;

        gl->BindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    p->upload_timer = gl_timer_create(p->gl);
    p->render_timer = gl_timer_create(p->gl);
    p->present_timer = gl_timer_create(p->gl);

    debug_check_gl(p, "after init_gl");
}

void gl_video_uninit(struct gl_video *p)
{
    if (!p)
        return;

    GL *gl = p->gl;

    uninit_video(p);

    gl_sc_destroy(p->sc);

    gl_vao_uninit(&p->vao);

    gl->DeleteTextures(1, &p->lut_3d_texture);

    gl_timer_free(p->upload_timer);
    gl_timer_free(p->render_timer);
    gl_timer_free(p->present_timer);

    mpgl_osd_destroy(p->osd);

    gl_set_debug_logger(gl, NULL);

    talloc_free(p);
}

void gl_video_set_gl_state(struct gl_video *p)
{
    // This resets certain important state to defaults.
    gl_video_unset_gl_state(p);
}

void gl_video_unset_gl_state(struct gl_video *p)
{
    GL *gl = p->gl;

    gl->ActiveTexture(GL_TEXTURE0);
    if (gl->mpgl_caps & MPGL_CAP_ROW_LENGTH)
        gl->PixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    gl->PixelStorei(GL_UNPACK_ALIGNMENT, 4);
}

void gl_video_reset(struct gl_video *p)
{
    gl_video_reset_surfaces(p);
}

bool gl_video_showing_interpolated_frame(struct gl_video *p)
{
    return p->is_interpolated;
}

// dest = src.<w> (always using 4 components)
static void packed_fmt_swizzle(char w[5], const struct packed_fmt_entry *fmt)
{
    for (int c = 0; c < 4; c++)
        w[c] = "rgba"[MPMAX(fmt->components[c] - 1, 0)];
    w[4] = '\0';
}

// Like gl_find_unorm_format(), but takes bits (not bytes), and if no fixed
// point format is available, return an unsigned integer format.
static const struct gl_format *find_plane_format(GL *gl, int bits, int n_channels)
{
    int bytes = (bits + 7) / 8;
    const struct gl_format *f = gl_find_unorm_format(gl, bytes, n_channels);
    if (f)
        return f;
    return gl_find_uint_format(gl, bytes, n_channels);
}

static void init_image_desc(struct gl_video *p, int fmt)
{
    p->image_desc = mp_imgfmt_get_desc(fmt);

    p->plane_count = p->image_desc.num_planes;
    p->is_yuv = p->image_desc.flags & MP_IMGFLAG_YUV;
    p->has_alpha = p->image_desc.flags & MP_IMGFLAG_ALPHA;
    p->use_integer_conversion = false;
    p->color_swizzle[0] = '\0';
    p->is_packed_yuv = fmt == IMGFMT_UYVY || fmt == IMGFMT_YUYV;
    p->hwdec_active = false;
}

// test_only=true checks if the format is supported
// test_only=false also initializes some rendering parameters accordingly
static bool init_format(struct gl_video *p, int fmt, bool test_only)
{
    struct GL *gl = p->gl;

    struct mp_imgfmt_desc desc = mp_imgfmt_get_desc(fmt);
    if (!desc.id)
        return false;

    if (desc.num_planes > 4)
        return false;

    const struct gl_format *plane_format[4] = {0};
    char color_swizzle[5] = "";
    const struct packed_fmt_entry *packed_format = {0};

    // YUV/planar formats
    if (desc.flags & (MP_IMGFLAG_YUV_P | MP_IMGFLAG_RGB_P)) {
        int bits = desc.component_bits;
        if ((desc.flags & MP_IMGFLAG_NE) && bits >= 8 && bits <= 16) {
            plane_format[0] = find_plane_format(gl, bits, 1);
            for (int n = 1; n < desc.num_planes; n++)
                plane_format[n] = plane_format[0];
            // RGB/planar
            if (desc.flags & MP_IMGFLAG_RGB_P)
                snprintf(color_swizzle, sizeof(color_swizzle), "brga");
            goto supported;
        }
    }

    // YUV/half-packed
    if (desc.flags & MP_IMGFLAG_YUV_NV) {
        int bits = desc.component_bits;
        if ((desc.flags & MP_IMGFLAG_NE) && bits >= 8 && bits <= 16) {
            plane_format[0] = find_plane_format(gl, bits, 1);
            plane_format[1] = find_plane_format(gl, bits, 2);
            if (desc.flags & MP_IMGFLAG_YUV_NV_SWAP)
                snprintf(color_swizzle, sizeof(color_swizzle), "rbga");
            goto supported;
        }
    }

    // XYZ (same organization as RGB packed, but requires conversion matrix)
    if (fmt == IMGFMT_XYZ12) {
        plane_format[0] = gl_find_unorm_format(gl, 2, 3);
        goto supported;
    }

    // Packed RGB(A) formats
    for (const struct packed_fmt_entry *e = mp_packed_formats; e->fmt; e++) {
        if (e->fmt == fmt) {
            int n_comp = desc.bytes[0] / e->component_size;
            plane_format[0] = gl_find_unorm_format(gl, e->component_size, n_comp);
            packed_format = e;
            goto supported;
        }
    }

    // Special formats for which OpenGL happens to have direct support.
    plane_format[0] = gl_find_special_format(gl, fmt);
    if (plane_format[0]) {
        // Packed YUV Apple formats color permutation
        if (plane_format[0]->format == GL_RGB_422_APPLE)
            snprintf(color_swizzle, sizeof(color_swizzle), "gbra");
        goto supported;
    }

    // Unsupported format
    return false;

supported:

    if (desc.component_bits > 8 && desc.component_bits < 16) {
        if (p->texture_16bit_depth < 16)
            return false;
    }

    int use_integer = -1;
    for (int n = 0; n < desc.num_planes; n++) {
        if (!plane_format[n])
            return false;
        int use_int_plane = !!gl_integer_format_to_base(plane_format[n]->format);
        if (use_integer < 0)
            use_integer = use_int_plane;
        if (use_integer != use_int_plane)
            return false; // mixed planes not supported
    }

    if (use_integer && p->forced_dumb_mode)
        return false;

    if (!test_only) {
        for (int n = 0; n < desc.num_planes; n++) {
            struct texplane *plane = &p->image.planes[n];
            const struct gl_format *format = plane_format[n];
            assert(format);
            plane->gl_format = format->format;
            plane->gl_internal_format = format->internal_format;
            plane->gl_type = format->type;
            plane->use_integer = use_integer;
            snprintf(plane->swizzle, sizeof(plane->swizzle), "rgba");
            if (packed_format)
                packed_fmt_swizzle(plane->swizzle, packed_format);
            if (plane->gl_format == GL_LUMINANCE_ALPHA)
                MPSWAP(char, plane->swizzle[1], plane->swizzle[3]);
        }

        init_image_desc(p, fmt);

        p->use_integer_conversion = use_integer;
        snprintf(p->color_swizzle, sizeof(p->color_swizzle), "%s", color_swizzle);
    }

    return true;
}

bool gl_video_check_format(struct gl_video *p, int mp_format)
{
    if (init_format(p, mp_format, true))
        return true;
    if (p->hwdec && p->hwdec->driver->imgfmt == mp_format)
        return true;
    return false;
}

void gl_video_config(struct gl_video *p, struct mp_image_params *params)
{
    mp_image_unrefp(&p->image.mpi);

    if (!mp_image_params_equal(&p->real_image_params, params)) {
        uninit_video(p);
        p->real_image_params = *params;
        p->image_params = *params;
        if (params->imgfmt)
            init_video(p);
    }

    gl_video_reset_surfaces(p);
}

void gl_video_set_osd_source(struct gl_video *p, struct osd_state *osd)
{
    mpgl_osd_destroy(p->osd);
    p->osd = NULL;
    p->osd_state = osd;
    reinit_osd(p);
}

struct gl_video *gl_video_init(GL *gl, struct mp_log *log, struct mpv_global *g)
{
    if (gl->version < 210 && gl->es < 200) {
        mp_err(log, "At least OpenGL 2.1 or OpenGL ES 2.0 required.\n");
        return NULL;
    }

    struct gl_video *p = talloc_ptrtype(NULL, p);
    *p = (struct gl_video) {
        .gl = gl,
        .global = g,
        .log = log,
        .cms = gl_lcms_init(p, log, g),
        .texture_16bit_depth = 16,
        .sc = gl_sc_create(gl, log),
    };
    set_options(p, NULL);
    for (int n = 0; n < SCALER_COUNT; n++)
        p->scaler[n] = (struct scaler){.index = n};
    gl_video_set_debug(p, true);
    init_gl(p);
    return p;
}

// Get static string for scaler shader. If "tscale" is set to true, the
// scaler must be a separable convolution filter.
static const char *handle_scaler_opt(const char *name, bool tscale)
{
    if (name && name[0]) {
        const struct filter_kernel *kernel = mp_find_filter_kernel(name);
        if (kernel && (!tscale || !kernel->polar))
                return kernel->f.name;

        for (const char *const *filter = tscale ? fixed_tscale_filters
                                                : fixed_scale_filters;
             *filter; filter++) {
            if (strcmp(*filter, name) == 0)
                return *filter;
        }
    }
    return NULL;
}

static void set_options(struct gl_video *p, struct gl_video_opts *src)
{
    talloc_free(p->opts_alloc);
    p->opts_alloc = m_sub_options_copy(p, &gl_video_conf, src);
    p->opts = *p->opts_alloc;
}

// Set the options, and possibly update the filter chain too.
// Note: assumes all options are valid and verified by the option parser.
void gl_video_set_options(struct gl_video *p, struct gl_video_opts *opts)
{
    set_options(p, opts);
    reinit_from_options(p);
}

static void reinit_from_options(struct gl_video *p)
{
    p->use_lut_3d = false;

    gl_lcms_set_options(p->cms, p->opts.icc_opts);
    p->use_lut_3d = gl_lcms_has_profile(p->cms);

    check_gl_features(p);
    uninit_rendering(p);
    gl_video_setup_hooks(p);
    reinit_osd(p);

    if (p->opts.interpolation && !p->global->opts->video_sync && !p->dsi_warned) {
        MP_WARN(p, "Interpolation now requires enabling display-sync mode.\n"
                   "E.g.: --video-sync=display-resample\n");
        p->dsi_warned = true;
    }
}

void gl_video_configure_queue(struct gl_video *p, struct vo *vo)
{
    int queue_size = 1;

    // Figure out an adequate size for the interpolation queue. The larger
    // the radius, the earlier we need to queue frames.
    if (p->opts.interpolation) {
        const struct filter_kernel *kernel =
            mp_find_filter_kernel(p->opts.scaler[SCALER_TSCALE].kernel.name);
        if (kernel) {
            double radius = kernel->f.radius;
            radius = radius > 0 ? radius : p->opts.scaler[SCALER_TSCALE].radius;
            queue_size += 1 + ceil(radius);
        } else {
            // Oversample case
            queue_size += 2;
        }
    }

    vo_set_queue_params(vo, 0, queue_size);
}

struct mp_csp_equalizer *gl_video_eq_ptr(struct gl_video *p)
{
    return &p->video_eq;
}

// Call when the mp_csp_equalizer returned by gl_video_eq_ptr() was changed.
void gl_video_eq_update(struct gl_video *p)
{
}

static int validate_scaler_opt(struct mp_log *log, const m_option_t *opt,
                               struct bstr name, struct bstr param)
{
    char s[20] = {0};
    int r = 1;
    bool tscale = bstr_equals0(name, "tscale");
    if (bstr_equals0(param, "help")) {
        r = M_OPT_EXIT - 1;
    } else {
        snprintf(s, sizeof(s), "%.*s", BSTR_P(param));
        if (!handle_scaler_opt(s, tscale))
            r = M_OPT_INVALID;
    }
    if (r < 1) {
        mp_info(log, "Available scalers:\n");
        for (const char *const *filter = tscale ? fixed_tscale_filters
                                                : fixed_scale_filters;
             *filter; filter++) {
            mp_info(log, "    %s\n", *filter);
        }
        for (int n = 0; mp_filter_kernels[n].f.name; n++) {
            if (!tscale || !mp_filter_kernels[n].polar)
                mp_info(log, "    %s\n", mp_filter_kernels[n].f.name);
        }
        if (s[0])
            mp_fatal(log, "No scaler named '%s' found!\n", s);
    }
    return r;
}

static int validate_window_opt(struct mp_log *log, const m_option_t *opt,
                               struct bstr name, struct bstr param)
{
    char s[20] = {0};
    int r = 1;
    if (bstr_equals0(param, "help")) {
        r = M_OPT_EXIT - 1;
    } else {
        snprintf(s, sizeof(s), "%.*s", BSTR_P(param));
        const struct filter_window *window = mp_find_filter_window(s);
        if (!window)
            r = M_OPT_INVALID;
    }
    if (r < 1) {
        mp_info(log, "Available windows:\n");
        for (int n = 0; mp_filter_windows[n].name; n++)
            mp_info(log, "    %s\n", mp_filter_windows[n].name);
        if (s[0])
            mp_fatal(log, "No window named '%s' found!\n", s);
    }
    return r;
}

float gl_video_scale_ambient_lux(float lmin, float lmax,
                                 float rmin, float rmax, float lux)
{
    assert(lmax > lmin);

    float num = (rmax - rmin) * (log10(lux) - log10(lmin));
    float den = log10(lmax) - log10(lmin);
    float result = num / den + rmin;

    // clamp the result
    float max = MPMAX(rmax, rmin);
    float min = MPMIN(rmax, rmin);
    return MPMAX(MPMIN(result, max), min);
}

