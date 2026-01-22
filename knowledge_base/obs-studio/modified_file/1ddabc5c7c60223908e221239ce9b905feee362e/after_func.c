{
	struct obs_core_video *video = &obs->video;

	if (resolution_close(video, width, height)) {
		return video->default_effect;
	} else {
		/* if the scale method couldn't be loaded, use either bicubic
		 * or bilinear by default */
		gs_effect_t *effect = get_scale_effect_internal(mix);
		if (!effect)
			effect = !!video->bicubic_effect
					 ? video->bicubic_effect
					 : video->default_effect;
		return effect;
	}
}

static const char *render_output_texture_name = "render_output_texture";
static inline gs_texture_t *
render_output_texture(struct obs_core_video_mix *mix)
{
	struct obs_core_video *video = &obs->video;
	gs_texture_t *texture = mix->render_texture;
	gs_texture_t *target = mix->output_texture;
	uint32_t width = gs_texture_get_width(target);
	uint32_t height = gs_texture_get_height(target);

	gs_effect_t *effect = get_scale_effect(mix, width, height);
	gs_technique_t *tech;

	if (video_output_get_format(mix->video) == VIDEO_FORMAT_RGBA) {
		tech = gs_effect_get_technique(effect, "DrawAlphaDivide");
	} else {
		if ((width == video->base_width) &&
		    (height == video->base_height))
			return texture;

		tech = gs_effect_get_technique(effect, "Draw");
	}

	profile_start(render_output_texture_name);

	gs_eparam_t *image = gs_effect_get_param_by_name(effect, "image");
	gs_eparam_t *bres =
		gs_effect_get_param_by_name(effect, "base_dimension");
	gs_eparam_t *bres_i =
		gs_effect_get_param_by_name(effect, "base_dimension_i");
	size_t passes, i;

	gs_set_render_target(target, NULL);
	set_render_size(width, height);

	if (bres) {
		struct vec2 base;
		vec2_set(&base, (float)video->base_width,
			 (float)video->base_height);
		gs_effect_set_vec2(bres, &base);
	}

	if (bres_i) {
		struct vec2 base_i;
		vec2_set(&base_i, 1.0f / (float)video->base_width,
			 1.0f / (float)video->base_height);
		gs_effect_set_vec2(bres_i, &base_i);
	}

