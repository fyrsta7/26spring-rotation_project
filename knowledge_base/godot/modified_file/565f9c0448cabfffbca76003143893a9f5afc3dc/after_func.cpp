void RendererSceneGIRD::VoxelGIInstance::update(bool p_update_light_instances, const Vector<RID> &p_light_instances, const PagedArray<RendererSceneRender::GeometryInstance *> &p_dynamic_objects, RendererSceneRenderRD *p_scene_render) {
	uint32_t data_version = storage->voxel_gi_get_data_version(probe);

	// (RE)CREATE IF NEEDED

	if (last_probe_data_version != data_version) {
		//need to re-create everything
		if (texture.is_valid()) {
			RD::get_singleton()->free(texture);
			RD::get_singleton()->free(write_buffer);
			mipmaps.clear();
		}

		for (int i = 0; i < dynamic_maps.size(); i++) {
			RD::get_singleton()->free(dynamic_maps[i].texture);
			RD::get_singleton()->free(dynamic_maps[i].depth);
		}

		dynamic_maps.clear();

		Vector3i octree_size = storage->voxel_gi_get_octree_size(probe);

		if (octree_size != Vector3i()) {
			//can create a 3D texture
			Vector<int> levels = storage->voxel_gi_get_level_counts(probe);

			RD::TextureFormat tf;
			tf.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
			tf.width = octree_size.x;
			tf.height = octree_size.y;
			tf.depth = octree_size.z;
			tf.texture_type = RD::TEXTURE_TYPE_3D;
			tf.mipmaps = levels.size();

			tf.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_CAN_COPY_TO_BIT;

			texture = RD::get_singleton()->texture_create(tf, RD::TextureView());

			RD::get_singleton()->texture_clear(texture, Color(0, 0, 0, 0), 0, levels.size(), 0, 1);

			{
				int total_elements = 0;
				for (int i = 0; i < levels.size(); i++) {
					total_elements += levels[i];
				}

				write_buffer = RD::get_singleton()->storage_buffer_create(total_elements * 16);
			}

			for (int i = 0; i < levels.size(); i++) {
				VoxelGIInstance::Mipmap mipmap;
				mipmap.texture = RD::get_singleton()->texture_create_shared_from_slice(RD::TextureView(), texture, 0, i, RD::TEXTURE_SLICE_3D);
				mipmap.level = levels.size() - i - 1;
				mipmap.cell_offset = 0;
				for (uint32_t j = 0; j < mipmap.level; j++) {
					mipmap.cell_offset += levels[j];
				}
				mipmap.cell_count = levels[mipmap.level];

				Vector<RD::Uniform> uniforms;
				{
					RD::Uniform u;
					u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
					u.binding = 1;
					u.ids.push_back(storage->voxel_gi_get_octree_buffer(probe));
					uniforms.push_back(u);
				}
				{
					RD::Uniform u;
					u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
					u.binding = 2;
					u.ids.push_back(storage->voxel_gi_get_data_buffer(probe));
					uniforms.push_back(u);
				}

				{
					RD::Uniform u;
					u.uniform_type = RD::UNIFORM_TYPE_STORAGE_BUFFER;
					u.binding = 4;
					u.ids.push_back(write_buffer);
					uniforms.push_back(u);
				}
				{
					RD::Uniform u;
					u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
					u.binding = 9;
					u.ids.push_back(storage->voxel_gi_get_sdf_texture(probe));
					uniforms.push_back(u);
				}
				{
					RD::Uniform u;
					u.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
					u.binding = 10;
					u.ids.push_back(storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED));
					uniforms.push_back(u);
				}

				{
					Vector<RD::Uniform> copy_uniforms = uniforms;
					if (i == 0) {
						{
							RD::Uniform u;
							u.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
							u.binding = 3;
							u.ids.push_back(gi->voxel_gi_lights_uniform);
							copy_uniforms.push_back(u);
						}

						mipmap.uniform_set = RD::get_singleton()->uniform_set_create(copy_uniforms, gi->voxel_gi_lighting_shader_version_shaders[VOXEL_GI_SHADER_VERSION_COMPUTE_LIGHT], 0);

						copy_uniforms = uniforms; //restore

						{
							RD::Uniform u;
							u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
							u.binding = 5;
							u.ids.push_back(texture);
							copy_uniforms.push_back(u);
						}
						mipmap.second_bounce_uniform_set = RD::get_singleton()->uniform_set_create(copy_uniforms, gi->voxel_gi_lighting_shader_version_shaders[VOXEL_GI_SHADER_VERSION_COMPUTE_SECOND_BOUNCE], 0);
					} else {
						mipmap.uniform_set = RD::get_singleton()->uniform_set_create(copy_uniforms, gi->voxel_gi_lighting_shader_version_shaders[VOXEL_GI_SHADER_VERSION_COMPUTE_MIPMAP], 0);
					}
				}

				{
					RD::Uniform u;
					u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
					u.binding = 5;
					u.ids.push_back(mipmap.texture);
					uniforms.push_back(u);
				}

				mipmap.write_uniform_set = RD::get_singleton()->uniform_set_create(uniforms, gi->voxel_gi_lighting_shader_version_shaders[VOXEL_GI_SHADER_VERSION_WRITE_TEXTURE], 0);

				mipmaps.push_back(mipmap);
			}

			{
				uint32_t dynamic_map_size = MAX(MAX(octree_size.x, octree_size.y), octree_size.z);
				uint32_t oversample = nearest_power_of_2_templated(4);
				int mipmap_index = 0;

				while (mipmap_index < mipmaps.size()) {
					VoxelGIInstance::DynamicMap dmap;

					if (oversample > 0) {
						dmap.size = dynamic_map_size * (1 << oversample);
						dmap.mipmap = -1;
						oversample--;
					} else {
						dmap.size = dynamic_map_size >> mipmap_index;
						dmap.mipmap = mipmap_index;
						mipmap_index++;
					}

					RD::TextureFormat dtf;
					dtf.width = dmap.size;
					dtf.height = dmap.size;
					dtf.format = RD::DATA_FORMAT_R16G16B16A16_SFLOAT;
					dtf.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT;

					if (dynamic_maps.size() == 0) {
						dtf.usage_bits |= RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;
					}
					dmap.texture = RD::get_singleton()->texture_create(dtf, RD::TextureView());

					if (dynamic_maps.size() == 0) {
						// Render depth for first one.
						// Use 16-bit depth when supported to improve performance.
						dtf.format = RD::get_singleton()->texture_is_format_supported_for_usage(RD::DATA_FORMAT_D16_UNORM, RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) ? RD::DATA_FORMAT_D16_UNORM : RD::DATA_FORMAT_X8_D24_UNORM_PACK32;
						dtf.usage_bits = RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
						dmap.fb_depth = RD::get_singleton()->texture_create(dtf, RD::TextureView());
					}

					//just use depth as-is
					dtf.format = RD::DATA_FORMAT_R32_SFLOAT;
					dtf.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;

					dmap.depth = RD::get_singleton()->texture_create(dtf, RD::TextureView());

					if (dynamic_maps.size() == 0) {
						dtf.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
						dtf.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT;
						dmap.albedo = RD::get_singleton()->texture_create(dtf, RD::TextureView());
						dmap.normal = RD::get_singleton()->texture_create(dtf, RD::TextureView());
						dmap.orm = RD::get_singleton()->texture_create(dtf, RD::TextureView());

						Vector<RID> fb;
						fb.push_back(dmap.albedo);
						fb.push_back(dmap.normal);
						fb.push_back(dmap.orm);
						fb.push_back(dmap.texture); //emission
						fb.push_back(dmap.depth);
						fb.push_back(dmap.fb_depth);

						dmap.fb = RD::get_singleton()->framebuffer_create(fb);

						{
							Vector<RD::Uniform> uniforms;
							{
								RD::Uniform u;
								u.uniform_type = RD::UNIFORM_TYPE_UNIFORM_BUFFER;
								u.binding = 3;
								u.ids.push_back(gi->voxel_gi_lights_uniform);
								uniforms.push_back(u);
							}

							{
								RD::Uniform u;
								u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
								u.binding = 5;
								u.ids.push_back(dmap.albedo);
								uniforms.push_back(u);
							}
							{
								RD::Uniform u;
								u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
								u.binding = 6;
								u.ids.push_back(dmap.normal);
								uniforms.push_back(u);
							}
							{
								RD::Uniform u;
								u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
								u.binding = 7;
								u.ids.push_back(dmap.orm);
								uniforms.push_back(u);
							}
							{
								RD::Uniform u;
								u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
								u.binding = 8;
								u.ids.push_back(dmap.fb_depth);
								uniforms.push_back(u);
							}
							{
								RD::Uniform u;
								u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
								u.binding = 9;
								u.ids.push_back(storage->voxel_gi_get_sdf_texture(probe));
								uniforms.push_back(u);
							}
							{
								RD::Uniform u;
								u.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
								u.binding = 10;
								u.ids.push_back(storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED));
								uniforms.push_back(u);
							}
							{
								RD::Uniform u;
								u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
								u.binding = 11;
								u.ids.push_back(dmap.texture);
								uniforms.push_back(u);
							}
							{
								RD::Uniform u;
								u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
								u.binding = 12;
								u.ids.push_back(dmap.depth);
								uniforms.push_back(u);
							}

							dmap.uniform_set = RD::get_singleton()->uniform_set_create(uniforms, gi->voxel_gi_lighting_shader_version_shaders[VOXEL_GI_SHADER_VERSION_DYNAMIC_OBJECT_LIGHTING], 0);
						}
					} else {
						bool plot = dmap.mipmap >= 0;
						bool write = dmap.mipmap < (mipmaps.size() - 1);

						Vector<RD::Uniform> uniforms;

						{
							RD::Uniform u;
							u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
							u.binding = 5;
							u.ids.push_back(dynamic_maps[dynamic_maps.size() - 1].texture);
							uniforms.push_back(u);
						}
						{
							RD::Uniform u;
							u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
							u.binding = 6;
							u.ids.push_back(dynamic_maps[dynamic_maps.size() - 1].depth);
							uniforms.push_back(u);
						}

						if (write) {
							{
								RD::Uniform u;
								u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
								u.binding = 7;
								u.ids.push_back(dmap.texture);
								uniforms.push_back(u);
							}
							{
								RD::Uniform u;
								u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
								u.binding = 8;
								u.ids.push_back(dmap.depth);
								uniforms.push_back(u);
							}
						}

						{
							RD::Uniform u;
							u.uniform_type = RD::UNIFORM_TYPE_TEXTURE;
							u.binding = 9;
							u.ids.push_back(storage->voxel_gi_get_sdf_texture(probe));
							uniforms.push_back(u);
						}
						{
							RD::Uniform u;
							u.uniform_type = RD::UNIFORM_TYPE_SAMPLER;
							u.binding = 10;
							u.ids.push_back(storage->sampler_rd_get_default(RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR_WITH_MIPMAPS, RS::CANVAS_ITEM_TEXTURE_REPEAT_DISABLED));
							uniforms.push_back(u);
						}

						if (plot) {
							{
								RD::Uniform u;
								u.uniform_type = RD::UNIFORM_TYPE_IMAGE;
								u.binding = 11;
								u.ids.push_back(mipmaps[dmap.mipmap].texture);
								uniforms.push_back(u);
							}
						}

						dmap.uniform_set = RD::get_singleton()->uniform_set_create(
								uniforms,
								gi->voxel_gi_lighting_shader_version_shaders[(write && plot) ? VOXEL_GI_SHADER_VERSION_DYNAMIC_SHRINK_WRITE_PLOT : (write ? VOXEL_GI_SHADER_VERSION_DYNAMIC_SHRINK_WRITE : VOXEL_GI_SHADER_VERSION_DYNAMIC_SHRINK_PLOT)],
								0);
					}

					dynamic_maps.push_back(dmap);
				}
			}
		}

		last_probe_data_version = data_version;
		p_update_light_instances = true; //just in case

		p_scene_render->_base_uniforms_changed();
	}

	// UDPDATE TIME

	if (has_dynamic_object_data) {
		//if it has dynamic object data, it needs to be cleared
		RD::get_singleton()->texture_clear(texture, Color(0, 0, 0, 0), 0, mipmaps.size(), 0, 1);
	}

	uint32_t light_count = 0;

	if (p_update_light_instances || p_dynamic_objects.size() > 0) {
		light_count = MIN(gi->voxel_gi_max_lights, (uint32_t)p_light_instances.size());

		{
			Transform3D to_cell = storage->voxel_gi_get_to_cell_xform(probe);
			Transform3D to_probe_xform = (transform * to_cell.affine_inverse()).affine_inverse();
			//update lights

			for (uint32_t i = 0; i < light_count; i++) {
				VoxelGILight &l = gi->voxel_gi_lights[i];
				RID light_instance = p_light_instances[i];
				RID light = p_scene_render->light_instance_get_base_light(light_instance);

				l.type = storage->light_get_type(light);
				if (l.type == RS::LIGHT_DIRECTIONAL && storage->light_directional_is_sky_only(light)) {
					light_count--;
					continue;
				}

				l.attenuation = storage->light_get_param(light, RS::LIGHT_PARAM_ATTENUATION);
				l.energy = storage->light_get_param(light, RS::LIGHT_PARAM_ENERGY) * storage->light_get_param(light, RS::LIGHT_PARAM_INDIRECT_ENERGY);
				l.radius = to_cell.basis.xform(Vector3(storage->light_get_param(light, RS::LIGHT_PARAM_RANGE), 0, 0)).length();
				Color color = storage->light_get_color(light).to_linear();
				l.color[0] = color.r;
				l.color[1] = color.g;
				l.color[2] = color.b;

				l.cos_spot_angle = Math::cos(Math::deg2rad(storage->light_get_param(light, RS::LIGHT_PARAM_SPOT_ANGLE)));
				l.inv_spot_attenuation = 1.0f / storage->light_get_param(light, RS::LIGHT_PARAM_SPOT_ATTENUATION);

				Transform3D xform = p_scene_render->light_instance_get_base_transform(light_instance);

				Vector3 pos = to_probe_xform.xform(xform.origin);
				Vector3 dir = to_probe_xform.basis.xform(-xform.basis.get_axis(2)).normalized();

				l.position[0] = pos.x;
				l.position[1] = pos.y;
				l.position[2] = pos.z;

				l.direction[0] = dir.x;
				l.direction[1] = dir.y;
				l.direction[2] = dir.z;

				l.has_shadow = storage->light_has_shadow(light);
			}

			RD::get_singleton()->buffer_update(gi->voxel_gi_lights_uniform, 0, sizeof(VoxelGILight) * light_count, gi->voxel_gi_lights);
		}
	}

	if (has_dynamic_object_data || p_update_light_instances || p_dynamic_objects.size()) {
		// PROCESS MIPMAPS
		if (mipmaps.size()) {
			//can update mipmaps

			Vector3i probe_size = storage->voxel_gi_get_octree_size(probe);

			VoxelGIPushConstant push_constant;

			push_constant.limits[0] = probe_size.x;
			push_constant.limits[1] = probe_size.y;
			push_constant.limits[2] = probe_size.z;
			push_constant.stack_size = mipmaps.size();
			push_constant.emission_scale = 1.0;
			push_constant.propagation = storage->voxel_gi_get_propagation(probe);
			push_constant.dynamic_range = storage->voxel_gi_get_dynamic_range(probe);
			push_constant.light_count = light_count;
			push_constant.aniso_strength = 0;

			/*		print_line("probe update to version " + itos(last_probe_version));
			print_line("propagation " + rtos(push_constant.propagation));
			print_line("dynrange " + rtos(push_constant.dynamic_range));
	*/
			RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();

			int passes;
			if (p_update_light_instances) {
				passes = storage->voxel_gi_is_using_two_bounces(probe) ? 2 : 1;
			} else {
				passes = 1; //only re-blitting is necessary
			}
			int wg_size = 64;
			int wg_limit_x = RD::get_singleton()->limit_get(RD::LIMIT_MAX_COMPUTE_WORKGROUP_COUNT_X);

			for (int pass = 0; pass < passes; pass++) {
				if (p_update_light_instances) {
					for (int i = 0; i < mipmaps.size(); i++) {
						if (i == 0) {
							RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, gi->voxel_gi_lighting_shader_version_pipelines[pass == 0 ? VOXEL_GI_SHADER_VERSION_COMPUTE_LIGHT : VOXEL_GI_SHADER_VERSION_COMPUTE_SECOND_BOUNCE]);
						} else if (i == 1) {
							RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, gi->voxel_gi_lighting_shader_version_pipelines[VOXEL_GI_SHADER_VERSION_COMPUTE_MIPMAP]);
						}

						if (pass == 1 || i > 0) {
							RD::get_singleton()->compute_list_add_barrier(compute_list); //wait til previous step is done
						}
						if (pass == 0 || i > 0) {
							RD::get_singleton()->compute_list_bind_uniform_set(compute_list, mipmaps[i].uniform_set, 0);
						} else {
							RD::get_singleton()->compute_list_bind_uniform_set(compute_list, mipmaps[i].second_bounce_uniform_set, 0);
						}

						push_constant.cell_offset = mipmaps[i].cell_offset;
						push_constant.cell_count = mipmaps[i].cell_count;

						int wg_todo = (mipmaps[i].cell_count - 1) / wg_size + 1;
						while (wg_todo) {
							int wg_count = MIN(wg_todo, wg_limit_x);
							RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(VoxelGIPushConstant));
							RD::get_singleton()->compute_list_dispatch(compute_list, wg_count, 1, 1);
							wg_todo -= wg_count;
							push_constant.cell_offset += wg_count * wg_size;
						}
					}

					RD::get_singleton()->compute_list_add_barrier(compute_list); //wait til previous step is done
				}

				RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, gi->voxel_gi_lighting_shader_version_pipelines[VOXEL_GI_SHADER_VERSION_WRITE_TEXTURE]);

				for (int i = 0; i < mipmaps.size(); i++) {
					RD::get_singleton()->compute_list_bind_uniform_set(compute_list, mipmaps[i].write_uniform_set, 0);

					push_constant.cell_offset = mipmaps[i].cell_offset;
					push_constant.cell_count = mipmaps[i].cell_count;

					int wg_todo = (mipmaps[i].cell_count - 1) / wg_size + 1;
					while (wg_todo) {
						int wg_count = MIN(wg_todo, wg_limit_x);
						RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(VoxelGIPushConstant));
						RD::get_singleton()->compute_list_dispatch(compute_list, wg_count, 1, 1);
						wg_todo -= wg_count;
						push_constant.cell_offset += wg_count * wg_size;
					}
				}
			}

			RD::get_singleton()->compute_list_end();
		}
	}

	has_dynamic_object_data = false; //clear until dynamic object data is used again

	if (p_dynamic_objects.size() && dynamic_maps.size()) {
		Vector3i octree_size = storage->voxel_gi_get_octree_size(probe);
		int multiplier = dynamic_maps[0].size / MAX(MAX(octree_size.x, octree_size.y), octree_size.z);

		Transform3D oversample_scale;
		oversample_scale.basis.scale(Vector3(multiplier, multiplier, multiplier));

		Transform3D to_cell = oversample_scale * storage->voxel_gi_get_to_cell_xform(probe);
		Transform3D to_world_xform = transform * to_cell.affine_inverse();
		Transform3D to_probe_xform = to_world_xform.affine_inverse();

		AABB probe_aabb(Vector3(), octree_size);

		//this could probably be better parallelized in compute..
		for (int i = 0; i < (int)p_dynamic_objects.size(); i++) {
			RendererSceneRender::GeometryInstance *instance = p_dynamic_objects[i];

			//transform aabb to voxel_gi
			AABB aabb = (to_probe_xform * p_scene_render->geometry_instance_get_transform(instance)).xform(p_scene_render->geometry_instance_get_aabb(instance));

			//this needs to wrap to grid resolution to avoid jitter
			//also extend margin a bit just in case
			Vector3i begin = aabb.position - Vector3i(1, 1, 1);
			Vector3i end = aabb.position + aabb.size + Vector3i(1, 1, 1);

			for (int j = 0; j < 3; j++) {
				if ((end[j] - begin[j]) & 1) {
					end[j]++; //for half extents split, it needs to be even
				}
				begin[j] = MAX(begin[j], 0);
				end[j] = MIN(end[j], octree_size[j] * multiplier);
			}

			//aabb = aabb.intersection(probe_aabb); //intersect
			aabb.position = begin;
			aabb.size = end - begin;

			//print_line("aabb: " + aabb);

			for (int j = 0; j < 6; j++) {
				//if (j != 0 && j != 3) {
				//	continue;
				//}
				static const Vector3 render_z[6] = {
					Vector3(1, 0, 0),
					Vector3(0, 1, 0),
					Vector3(0, 0, 1),
					Vector3(-1, 0, 0),
					Vector3(0, -1, 0),
					Vector3(0, 0, -1),
				};
				static const Vector3 render_up[6] = {
					Vector3(0, 1, 0),
					Vector3(0, 0, 1),
					Vector3(0, 1, 0),
					Vector3(0, 1, 0),
					Vector3(0, 0, 1),
					Vector3(0, 1, 0),
				};

				Vector3 render_dir = render_z[j];
				Vector3 up_dir = render_up[j];

				Vector3 center = aabb.get_center();
				Transform3D xform;
				xform.set_look_at(center - aabb.size * 0.5 * render_dir, center, up_dir);

				Vector3 x_dir = xform.basis.get_axis(0).abs();
				int x_axis = int(Vector3(0, 1, 2).dot(x_dir));
				Vector3 y_dir = xform.basis.get_axis(1).abs();
				int y_axis = int(Vector3(0, 1, 2).dot(y_dir));
				Vector3 z_dir = -xform.basis.get_axis(2);
				int z_axis = int(Vector3(0, 1, 2).dot(z_dir.abs()));

				Rect2i rect(aabb.position[x_axis], aabb.position[y_axis], aabb.size[x_axis], aabb.size[y_axis]);
				bool x_flip = bool(Vector3(1, 1, 1).dot(xform.basis.get_axis(0)) < 0);
				bool y_flip = bool(Vector3(1, 1, 1).dot(xform.basis.get_axis(1)) < 0);
				bool z_flip = bool(Vector3(1, 1, 1).dot(xform.basis.get_axis(2)) > 0);

				CameraMatrix cm;
				cm.set_orthogonal(-rect.size.width / 2, rect.size.width / 2, -rect.size.height / 2, rect.size.height / 2, 0.0001, aabb.size[z_axis]);

				if (p_scene_render->cull_argument.size() == 0) {
					p_scene_render->cull_argument.push_back(nullptr);
				}
				p_scene_render->cull_argument[0] = instance;

				p_scene_render->_render_material(to_world_xform * xform, cm, true, p_scene_render->cull_argument, dynamic_maps[0].fb, Rect2i(Vector2i(), rect.size));

				VoxelGIDynamicPushConstant push_constant;
				memset(&push_constant, 0, sizeof(VoxelGIDynamicPushConstant));
				push_constant.limits[0] = octree_size.x;
				push_constant.limits[1] = octree_size.y;
				push_constant.limits[2] = octree_size.z;
				push_constant.light_count = p_light_instances.size();
				push_constant.x_dir[0] = x_dir[0];
				push_constant.x_dir[1] = x_dir[1];
				push_constant.x_dir[2] = x_dir[2];
				push_constant.y_dir[0] = y_dir[0];
				push_constant.y_dir[1] = y_dir[1];
				push_constant.y_dir[2] = y_dir[2];
				push_constant.z_dir[0] = z_dir[0];
				push_constant.z_dir[1] = z_dir[1];
				push_constant.z_dir[2] = z_dir[2];
				push_constant.z_base = xform.origin[z_axis];
				push_constant.z_sign = (z_flip ? -1.0 : 1.0);
				push_constant.pos_multiplier = float(1.0) / multiplier;
				push_constant.dynamic_range = storage->voxel_gi_get_dynamic_range(probe);
				push_constant.flip_x = x_flip;
				push_constant.flip_y = y_flip;
				push_constant.rect_pos[0] = rect.position[0];
				push_constant.rect_pos[1] = rect.position[1];
				push_constant.rect_size[0] = rect.size[0];
				push_constant.rect_size[1] = rect.size[1];
				push_constant.prev_rect_ofs[0] = 0;
				push_constant.prev_rect_ofs[1] = 0;
				push_constant.prev_rect_size[0] = 0;
				push_constant.prev_rect_size[1] = 0;
				push_constant.on_mipmap = false;
				push_constant.propagation = storage->voxel_gi_get_propagation(probe);
				push_constant.pad[0] = 0;
				push_constant.pad[1] = 0;
				push_constant.pad[2] = 0;

				//process lighting
				RD::ComputeListID compute_list = RD::get_singleton()->compute_list_begin();
				RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, gi->voxel_gi_lighting_shader_version_pipelines[VOXEL_GI_SHADER_VERSION_DYNAMIC_OBJECT_LIGHTING]);
				RD::get_singleton()->compute_list_bind_uniform_set(compute_list, dynamic_maps[0].uniform_set, 0);
				RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(VoxelGIDynamicPushConstant));
				RD::get_singleton()->compute_list_dispatch(compute_list, (rect.size.x - 1) / 8 + 1, (rect.size.y - 1) / 8 + 1, 1);
				//print_line("rect: " + itos(i) + ": " + rect);

				for (int k = 1; k < dynamic_maps.size(); k++) {
					// enlarge the rect if needed so all pixels fit when downscaled,
					// this ensures downsampling is smooth and optimal because no pixels are left behind

					//x
					if (rect.position.x & 1) {
						rect.size.x++;
						push_constant.prev_rect_ofs[0] = 1; //this is used to ensure reading is also optimal
					} else {
						push_constant.prev_rect_ofs[0] = 0;
					}
					if (rect.size.x & 1) {
						rect.size.x++;
					}

					rect.position.x >>= 1;
					rect.size.x = MAX(1, rect.size.x >> 1);

					//y
					if (rect.position.y & 1) {
						rect.size.y++;
						push_constant.prev_rect_ofs[1] = 1;
					} else {
						push_constant.prev_rect_ofs[1] = 0;
					}
					if (rect.size.y & 1) {
						rect.size.y++;
					}

					rect.position.y >>= 1;
					rect.size.y = MAX(1, rect.size.y >> 1);

					//shrink limits to ensure plot does not go outside map
					if (dynamic_maps[k].mipmap > 0) {
						for (int l = 0; l < 3; l++) {
							push_constant.limits[l] = MAX(1, push_constant.limits[l] >> 1);
						}
					}

					//print_line("rect: " + itos(i) + ": " + rect);
					push_constant.rect_pos[0] = rect.position[0];
					push_constant.rect_pos[1] = rect.position[1];
					push_constant.prev_rect_size[0] = push_constant.rect_size[0];
					push_constant.prev_rect_size[1] = push_constant.rect_size[1];
					push_constant.rect_size[0] = rect.size[0];
					push_constant.rect_size[1] = rect.size[1];
					push_constant.on_mipmap = dynamic_maps[k].mipmap > 0;

					RD::get_singleton()->compute_list_add_barrier(compute_list);

					if (dynamic_maps[k].mipmap < 0) {
						RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, gi->voxel_gi_lighting_shader_version_pipelines[VOXEL_GI_SHADER_VERSION_DYNAMIC_SHRINK_WRITE]);
					} else if (k < dynamic_maps.size() - 1) {
						RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, gi->voxel_gi_lighting_shader_version_pipelines[VOXEL_GI_SHADER_VERSION_DYNAMIC_SHRINK_WRITE_PLOT]);
					} else {
						RD::get_singleton()->compute_list_bind_compute_pipeline(compute_list, gi->voxel_gi_lighting_shader_version_pipelines[VOXEL_GI_SHADER_VERSION_DYNAMIC_SHRINK_PLOT]);
					}
					RD::get_singleton()->compute_list_bind_uniform_set(compute_list, dynamic_maps[k].uniform_set, 0);
					RD::get_singleton()->compute_list_set_push_constant(compute_list, &push_constant, sizeof(VoxelGIDynamicPushConstant));
					RD::get_singleton()->compute_list_dispatch(compute_list, (rect.size.x - 1) / 8 + 1, (rect.size.y - 1) / 8 + 1, 1);
				}

				RD::get_singleton()->compute_list_end();
			}
		}

		has_dynamic_object_data = true; //clear until dynamic object data is used again
	}
