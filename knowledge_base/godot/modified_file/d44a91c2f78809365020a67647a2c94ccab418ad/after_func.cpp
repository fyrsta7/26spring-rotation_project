void SceneShaderForwardClustered::ShaderData::set_code(const String &p_code) {
	//compile

	code = p_code;
	valid = false;
	ubo_size = 0;
	uniforms.clear();
	uses_screen_texture = false;

	if (code.is_empty()) {
		return; //just invalid, but no error
	}

	ShaderCompiler::GeneratedCode gen_code;

	int blend_mode = BLEND_MODE_MIX;
	int depth_testi = DEPTH_TEST_ENABLED;
	int alpha_antialiasing_mode = ALPHA_ANTIALIASING_OFF;
	int cull_modei = CULL_BACK;

	uses_point_size = false;
	uses_alpha = false;
	uses_alpha_clip = false;
	uses_blend_alpha = false;
	uses_depth_pre_pass = false;
	uses_discard = false;
	uses_roughness = false;
	uses_normal = false;
	bool wireframe = false;

	unshaded = false;
	uses_vertex = false;
	uses_position = false;
	uses_sss = false;
	uses_transmittance = false;
	uses_screen_texture = false;
	uses_depth_texture = false;
	uses_normal_texture = false;
	uses_time = false;
	writes_modelview_or_projection = false;
	uses_world_coordinates = false;
	uses_particle_trails = false;

	int depth_drawi = DEPTH_DRAW_OPAQUE;

	ShaderCompiler::IdentifierActions actions;
	actions.entry_point_stages["vertex"] = ShaderCompiler::STAGE_VERTEX;
	actions.entry_point_stages["fragment"] = ShaderCompiler::STAGE_FRAGMENT;
	actions.entry_point_stages["light"] = ShaderCompiler::STAGE_FRAGMENT;

	actions.render_mode_values["blend_add"] = Pair<int *, int>(&blend_mode, BLEND_MODE_ADD);
	actions.render_mode_values["blend_mix"] = Pair<int *, int>(&blend_mode, BLEND_MODE_MIX);
	actions.render_mode_values["blend_sub"] = Pair<int *, int>(&blend_mode, BLEND_MODE_SUB);
	actions.render_mode_values["blend_mul"] = Pair<int *, int>(&blend_mode, BLEND_MODE_MUL);

	actions.render_mode_values["alpha_to_coverage"] = Pair<int *, int>(&alpha_antialiasing_mode, ALPHA_ANTIALIASING_ALPHA_TO_COVERAGE);
	actions.render_mode_values["alpha_to_coverage_and_one"] = Pair<int *, int>(&alpha_antialiasing_mode, ALPHA_ANTIALIASING_ALPHA_TO_COVERAGE_AND_TO_ONE);

	actions.render_mode_values["depth_draw_never"] = Pair<int *, int>(&depth_drawi, DEPTH_DRAW_DISABLED);
	actions.render_mode_values["depth_draw_opaque"] = Pair<int *, int>(&depth_drawi, DEPTH_DRAW_OPAQUE);
	actions.render_mode_values["depth_draw_always"] = Pair<int *, int>(&depth_drawi, DEPTH_DRAW_ALWAYS);

	actions.render_mode_values["depth_test_disabled"] = Pair<int *, int>(&depth_testi, DEPTH_TEST_DISABLED);

	actions.render_mode_values["cull_disabled"] = Pair<int *, int>(&cull_modei, CULL_DISABLED);
	actions.render_mode_values["cull_front"] = Pair<int *, int>(&cull_modei, CULL_FRONT);
	actions.render_mode_values["cull_back"] = Pair<int *, int>(&cull_modei, CULL_BACK);

	actions.render_mode_flags["unshaded"] = &unshaded;
	actions.render_mode_flags["wireframe"] = &wireframe;
	actions.render_mode_flags["particle_trails"] = &uses_particle_trails;

	actions.usage_flag_pointers["ALPHA"] = &uses_alpha;
	actions.usage_flag_pointers["ALPHA_SCISSOR_THRESHOLD"] = &uses_alpha_clip;
	// Use alpha clip pipeline for alpha hash/dither.
	// This prevents sorting issues inherent to alpha blending and allows such materials to cast shadows.
	actions.usage_flag_pointers["ALPHA_HASH_SCALE"] = &uses_alpha_clip;
	actions.render_mode_flags["depth_prepass_alpha"] = &uses_depth_pre_pass;

	actions.usage_flag_pointers["SSS_STRENGTH"] = &uses_sss;
	actions.usage_flag_pointers["SSS_TRANSMITTANCE_DEPTH"] = &uses_transmittance;

	actions.usage_flag_pointers["SCREEN_TEXTURE"] = &uses_screen_texture;
	actions.usage_flag_pointers["DEPTH_TEXTURE"] = &uses_depth_texture;
	actions.usage_flag_pointers["NORMAL_TEXTURE"] = &uses_normal_texture;
	actions.usage_flag_pointers["DISCARD"] = &uses_discard;
	actions.usage_flag_pointers["TIME"] = &uses_time;
	actions.usage_flag_pointers["ROUGHNESS"] = &uses_roughness;
	actions.usage_flag_pointers["NORMAL"] = &uses_normal;
	actions.usage_flag_pointers["NORMAL_MAP"] = &uses_normal;

	actions.usage_flag_pointers["POINT_SIZE"] = &uses_point_size;
	actions.usage_flag_pointers["POINT_COORD"] = &uses_point_size;

	actions.write_flag_pointers["MODELVIEW_MATRIX"] = &writes_modelview_or_projection;
	actions.write_flag_pointers["PROJECTION_MATRIX"] = &writes_modelview_or_projection;
	actions.write_flag_pointers["VERTEX"] = &uses_vertex;
	actions.write_flag_pointers["POSITION"] = &uses_position;

	actions.uniforms = &uniforms;

	SceneShaderForwardClustered *shader_singleton = (SceneShaderForwardClustered *)SceneShaderForwardClustered::singleton;
	Error err = shader_singleton->compiler.compile(RS::SHADER_SPATIAL, code, &actions, path, gen_code);
	ERR_FAIL_COND_MSG(err != OK, "Shader compilation failed.");

	if (version.is_null()) {
		version = shader_singleton->shader.version_create();
	}

	depth_draw = DepthDraw(depth_drawi);
	depth_test = DepthTest(depth_testi);
	cull_mode = Cull(cull_modei);
	uses_screen_texture_mipmaps = gen_code.uses_screen_texture_mipmaps;
	uses_vertex_time = gen_code.uses_vertex_time;
	uses_fragment_time = gen_code.uses_fragment_time;

#if 0
	print_line("**compiling shader:");
	print_line("**defines:\n");
	for (int i = 0; i < gen_code.defines.size(); i++) {
		print_line(gen_code.defines[i]);
	}

	HashMap<String, String>::Iterator el = gen_code.code.begin();
	while (el) {
		print_line("\n**code " + el->key + ":\n" + el->value);
		++el;
	}

	print_line("\n**uniforms:\n" + gen_code.uniforms);
	print_line("\n**vertex_globals:\n" + gen_code.stage_globals[ShaderCompiler::STAGE_VERTEX]);
	print_line("\n**fragment_globals:\n" + gen_code.stage_globals[ShaderCompiler::STAGE_FRAGMENT]);
#endif
	shader_singleton->shader.version_set_code(version, gen_code.code, gen_code.uniforms, gen_code.stage_globals[ShaderCompiler::STAGE_VERTEX], gen_code.stage_globals[ShaderCompiler::STAGE_FRAGMENT], gen_code.defines);
	ERR_FAIL_COND(!shader_singleton->shader.version_is_valid(version));

	ubo_size = gen_code.uniform_total_size;
	ubo_offsets = gen_code.uniform_offsets;
	texture_uniforms = gen_code.texture_uniforms;

	//blend modes

	// if any form of Alpha Antialiasing is enabled, set the blend mode to alpha to coverage
	if (alpha_antialiasing_mode != ALPHA_ANTIALIASING_OFF) {
		blend_mode = BLEND_MODE_ALPHA_TO_COVERAGE;
	}

	RD::PipelineColorBlendState::Attachment blend_attachment;

	switch (blend_mode) {
		case BLEND_MODE_MIX: {
			blend_attachment.enable_blend = true;
			blend_attachment.alpha_blend_op = RD::BLEND_OP_ADD;
			blend_attachment.color_blend_op = RD::BLEND_OP_ADD;
			blend_attachment.src_color_blend_factor = RD::BLEND_FACTOR_SRC_ALPHA;
			blend_attachment.dst_color_blend_factor = RD::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
			blend_attachment.src_alpha_blend_factor = RD::BLEND_FACTOR_ONE;
			blend_attachment.dst_alpha_blend_factor = RD::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;

		} break;
		case BLEND_MODE_ADD: {
			blend_attachment.enable_blend = true;
			blend_attachment.alpha_blend_op = RD::BLEND_OP_ADD;
			blend_attachment.color_blend_op = RD::BLEND_OP_ADD;
			blend_attachment.src_color_blend_factor = RD::BLEND_FACTOR_SRC_ALPHA;
			blend_attachment.dst_color_blend_factor = RD::BLEND_FACTOR_ONE;
			blend_attachment.src_alpha_blend_factor = RD::BLEND_FACTOR_SRC_ALPHA;
			blend_attachment.dst_alpha_blend_factor = RD::BLEND_FACTOR_ONE;
			uses_blend_alpha = true; //force alpha used because of blend

		} break;
		case BLEND_MODE_SUB: {
			blend_attachment.enable_blend = true;
			blend_attachment.alpha_blend_op = RD::BLEND_OP_SUBTRACT;
			blend_attachment.color_blend_op = RD::BLEND_OP_SUBTRACT;
			blend_attachment.src_color_blend_factor = RD::BLEND_FACTOR_SRC_ALPHA;
			blend_attachment.dst_color_blend_factor = RD::BLEND_FACTOR_ONE;
			blend_attachment.src_alpha_blend_factor = RD::BLEND_FACTOR_SRC_ALPHA;
			blend_attachment.dst_alpha_blend_factor = RD::BLEND_FACTOR_ONE;
			uses_blend_alpha = true; //force alpha used because of blend

		} break;
		case BLEND_MODE_MUL: {
			blend_attachment.enable_blend = true;
			blend_attachment.alpha_blend_op = RD::BLEND_OP_ADD;
			blend_attachment.color_blend_op = RD::BLEND_OP_ADD;
			blend_attachment.src_color_blend_factor = RD::BLEND_FACTOR_DST_COLOR;
			blend_attachment.dst_color_blend_factor = RD::BLEND_FACTOR_ZERO;
			blend_attachment.src_alpha_blend_factor = RD::BLEND_FACTOR_DST_ALPHA;
			blend_attachment.dst_alpha_blend_factor = RD::BLEND_FACTOR_ZERO;
			uses_blend_alpha = true; //force alpha used because of blend
		} break;
		case BLEND_MODE_ALPHA_TO_COVERAGE: {
			blend_attachment.enable_blend = true;
			blend_attachment.alpha_blend_op = RD::BLEND_OP_ADD;
			blend_attachment.color_blend_op = RD::BLEND_OP_ADD;
			blend_attachment.src_color_blend_factor = RD::BLEND_FACTOR_SRC_ALPHA;
			blend_attachment.dst_color_blend_factor = RD::BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
			blend_attachment.src_alpha_blend_factor = RD::BLEND_FACTOR_ONE;
			blend_attachment.dst_alpha_blend_factor = RD::BLEND_FACTOR_ZERO;
		}
	}

	// Color pass -> attachment 0: Color/Diffuse, attachment 1: Separate Specular, attachment 2: Motion Vectors
	RD::PipelineColorBlendState blend_state_color_blend;
	blend_state_color_blend.attachments = { blend_attachment, RD::PipelineColorBlendState::Attachment(), RD::PipelineColorBlendState::Attachment() };
	RD::PipelineColorBlendState blend_state_color_opaque = RD::PipelineColorBlendState::create_disabled(3);
	RD::PipelineColorBlendState blend_state_depth_normal_roughness = RD::PipelineColorBlendState::create_disabled(1);
	RD::PipelineColorBlendState blend_state_depth_normal_roughness_giprobe = RD::PipelineColorBlendState::create_disabled(2);

	//update pipelines

	RD::PipelineDepthStencilState depth_stencil_state;

	if (depth_test != DEPTH_TEST_DISABLED) {
		depth_stencil_state.enable_depth_test = true;
		depth_stencil_state.depth_compare_operator = RD::COMPARE_OP_LESS_OR_EQUAL;
		depth_stencil_state.enable_depth_write = depth_draw != DEPTH_DRAW_DISABLED ? true : false;
	}
	bool depth_pre_pass_enabled = bool(GLOBAL_GET("rendering/driver/depth_prepass/enable"));

	for (int i = 0; i < CULL_VARIANT_MAX; i++) {
		RD::PolygonCullMode cull_mode_rd_table[CULL_VARIANT_MAX][3] = {
			{ RD::POLYGON_CULL_DISABLED, RD::POLYGON_CULL_FRONT, RD::POLYGON_CULL_BACK },
			{ RD::POLYGON_CULL_DISABLED, RD::POLYGON_CULL_BACK, RD::POLYGON_CULL_FRONT },
			{ RD::POLYGON_CULL_DISABLED, RD::POLYGON_CULL_DISABLED, RD::POLYGON_CULL_DISABLED }
		};

		RD::PolygonCullMode cull_mode_rd = cull_mode_rd_table[i][cull_mode];

		for (int j = 0; j < RS::PRIMITIVE_MAX; j++) {
			RD::RenderPrimitive primitive_rd_table[RS::PRIMITIVE_MAX] = {
				RD::RENDER_PRIMITIVE_POINTS,
				RD::RENDER_PRIMITIVE_LINES,
				RD::RENDER_PRIMITIVE_LINESTRIPS,
				RD::RENDER_PRIMITIVE_TRIANGLES,
				RD::RENDER_PRIMITIVE_TRIANGLE_STRIPS,
			};

			RD::RenderPrimitive primitive_rd = uses_point_size ? RD::RENDER_PRIMITIVE_POINTS : primitive_rd_table[j];

			for (int k = 0; k < PIPELINE_VERSION_MAX; k++) {
				ShaderVersion shader_version;
				static const ShaderVersion shader_version_table[PIPELINE_VERSION_MAX] = {
					SHADER_VERSION_DEPTH_PASS,
					SHADER_VERSION_DEPTH_PASS_DP,
					SHADER_VERSION_DEPTH_PASS_WITH_NORMAL_AND_ROUGHNESS,
					SHADER_VERSION_DEPTH_PASS_WITH_NORMAL_AND_ROUGHNESS_AND_VOXEL_GI,
					SHADER_VERSION_DEPTH_PASS_WITH_MATERIAL,
					SHADER_VERSION_DEPTH_PASS_WITH_SDF,
					SHADER_VERSION_DEPTH_PASS_MULTIVIEW,
					SHADER_VERSION_DEPTH_PASS_WITH_NORMAL_AND_ROUGHNESS_MULTIVIEW,
					SHADER_VERSION_DEPTH_PASS_WITH_NORMAL_AND_ROUGHNESS_AND_VOXEL_GI_MULTIVIEW,
					SHADER_VERSION_COLOR_PASS,
				};

				shader_version = shader_version_table[k];

				if (!static_cast<SceneShaderForwardClustered *>(singleton)->shader.is_variant_enabled(shader_version)) {
					continue;
				}
				RD::PipelineRasterizationState raster_state;
				raster_state.cull_mode = cull_mode_rd;
				raster_state.wireframe = wireframe;

				if (k == PIPELINE_VERSION_COLOR_PASS) {
					for (int l = 0; l < PIPELINE_COLOR_PASS_FLAG_COUNT; l++) {
						if (!shader_singleton->valid_color_pass_pipelines.has(l)) {
							continue;
						}

						RD::PipelineDepthStencilState depth_stencil = depth_stencil_state;
						if (depth_pre_pass_enabled && casts_shadows()) {
							// We already have a depth from the depth pre-pass, there is no need to write it again.
							// In addition we can use COMPARE_OP_EQUAL instead of COMPARE_OP_LESS_OR_EQUAL.
							// This way we can use the early depth test to discard transparent fragments before the fragment shader even starts.
							depth_stencil.depth_compare_operator = RD::COMPARE_OP_EQUAL;
							depth_stencil.enable_depth_write = false;
						}

						RD::PipelineColorBlendState blend_state;
						RD::PipelineMultisampleState multisample_state;

						int shader_flags = 0;
						if (l & PIPELINE_COLOR_PASS_FLAG_TRANSPARENT) {
							if (alpha_antialiasing_mode == ALPHA_ANTIALIASING_ALPHA_TO_COVERAGE) {
								multisample_state.enable_alpha_to_coverage = true;
							} else if (alpha_antialiasing_mode == ALPHA_ANTIALIASING_ALPHA_TO_COVERAGE_AND_TO_ONE) {
								multisample_state.enable_alpha_to_coverage = true;
								multisample_state.enable_alpha_to_one = true;
							}

							blend_state = blend_state_color_blend;

							if (depth_draw == DEPTH_DRAW_OPAQUE) {
								depth_stencil.enable_depth_write = false; //alpha does not draw depth
							}
						} else {
							blend_state = blend_state_color_opaque;

							if (l & PIPELINE_COLOR_PASS_FLAG_SEPARATE_SPECULAR) {
								shader_flags |= SHADER_COLOR_PASS_FLAG_SEPARATE_SPECULAR;
							}
						}

						if (l & PIPELINE_COLOR_PASS_FLAG_MOTION_VECTORS) {
							shader_flags |= SHADER_COLOR_PASS_FLAG_MOTION_VECTORS;
						}

						if (l & PIPELINE_COLOR_PASS_FLAG_LIGHTMAP) {
							shader_flags |= SHADER_COLOR_PASS_FLAG_LIGHTMAP;
						}

						if (l & PIPELINE_COLOR_PASS_FLAG_MULTIVIEW) {
							shader_flags |= SHADER_COLOR_PASS_FLAG_MULTIVIEW;
						}

						int variant = shader_version + shader_flags;
						RID shader_variant = shader_singleton->shader.version_get_shader(version, variant);
						color_pipelines[i][j][l].setup(shader_variant, primitive_rd, raster_state, multisample_state, depth_stencil, blend_state, 0, singleton->default_specialization_constants);
					}
				} else {
					RD::PipelineColorBlendState blend_state;
					RD::PipelineDepthStencilState depth_stencil = depth_stencil_state;
					RD::PipelineMultisampleState multisample_state;

					if (k == PIPELINE_VERSION_DEPTH_PASS || k == PIPELINE_VERSION_DEPTH_PASS_DP || k == PIPELINE_VERSION_DEPTH_PASS_MULTIVIEW) {
						//none, leave empty
					} else if (k == PIPELINE_VERSION_DEPTH_PASS_WITH_NORMAL_AND_ROUGHNESS || k == PIPELINE_VERSION_DEPTH_PASS_WITH_NORMAL_AND_ROUGHNESS_MULTIVIEW) {
						blend_state = blend_state_depth_normal_roughness;
					} else if (k == PIPELINE_VERSION_DEPTH_PASS_WITH_NORMAL_AND_ROUGHNESS_AND_VOXEL_GI || k == PIPELINE_VERSION_DEPTH_PASS_WITH_NORMAL_AND_ROUGHNESS_AND_VOXEL_GI_MULTIVIEW) {
						blend_state = blend_state_depth_normal_roughness_giprobe;
					} else if (k == PIPELINE_VERSION_DEPTH_PASS_WITH_MATERIAL) {
						blend_state = RD::PipelineColorBlendState::create_disabled(5); //writes to normal and roughness in opaque way
					} else if (k == PIPELINE_VERSION_DEPTH_PASS_WITH_SDF) {
						blend_state = RD::PipelineColorBlendState(); //no color targets for SDF
					}

					RID shader_variant = shader_singleton->shader.version_get_shader(version, shader_version);
					pipelines[i][j][k].setup(shader_variant, primitive_rd, raster_state, multisample_state, depth_stencil, blend_state, 0, singleton->default_specialization_constants);
				}
			}
		}
	}

	valid = true;
}
