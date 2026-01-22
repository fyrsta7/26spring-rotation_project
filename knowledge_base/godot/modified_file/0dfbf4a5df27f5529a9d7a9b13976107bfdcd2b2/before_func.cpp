void RendererCanvasRenderRD::_render_batch(RD::DrawListID p_draw_list, CanvasShaderData *p_shader_data, RenderingDevice::FramebufferFormatID p_framebuffer_format, Light *p_lights, Batch const *p_batch, RenderingMethod::RenderInfo *r_render_info) {
	{
		RIDSetKey key(
				p_batch->tex_info->state,
				state.canvas_instance_data_buffers[state.current_data_buffer_index].instance_buffers[p_batch->instance_buffer_index]);

		const RID *uniform_set = rid_set_to_uniform_set.getptr(key);
		if (uniform_set == nullptr) {
			state.batch_texture_uniforms.write[0] = RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 0, p_batch->tex_info->diffuse);
			state.batch_texture_uniforms.write[1] = RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 1, p_batch->tex_info->normal);
			state.batch_texture_uniforms.write[2] = RD::Uniform(RD::UNIFORM_TYPE_TEXTURE, 2, p_batch->tex_info->specular);
			state.batch_texture_uniforms.write[3] = RD::Uniform(RD::UNIFORM_TYPE_SAMPLER, 3, p_batch->tex_info->sampler);
			state.batch_texture_uniforms.write[4] = RD::Uniform(RD::UNIFORM_TYPE_STORAGE_BUFFER, 4, state.canvas_instance_data_buffers[state.current_data_buffer_index].instance_buffers[p_batch->instance_buffer_index]);

			RID rid = RD::get_singleton()->uniform_set_create(state.batch_texture_uniforms, shader.default_version_rd_shader, BATCH_UNIFORM_SET);
			ERR_FAIL_COND_MSG(rid.is_null(), "Failed to create uniform set for batch.");

			const RIDCache::Pair *iter = rid_set_to_uniform_set.insert(key, rid);
			uniform_set = &iter->data;
			RD::get_singleton()->uniform_set_set_invalidation_callback(rid, RendererCanvasRenderRD::_uniform_set_invalidation_callback, (void *)&iter->key);
		}

		if (state.current_batch_uniform_set != *uniform_set) {
			state.current_batch_uniform_set = *uniform_set;
			RD::get_singleton()->draw_list_bind_uniform_set(p_draw_list, *uniform_set, BATCH_UNIFORM_SET);
		}
	}
	PushConstant push_constant;
	push_constant.base_instance_index = p_batch->start;
	push_constant.specular_shininess = p_batch->tex_info->specular_shininess;
	push_constant.batch_flags = p_batch->tex_info->flags | p_batch->flags;

	RID pipeline;
	PipelineKey pipeline_key;
	pipeline_key.framebuffer_format_id = p_framebuffer_format;
	pipeline_key.variant = p_batch->shader_variant;
	pipeline_key.render_primitive = p_batch->render_primitive;
	pipeline_key.shader_specialization.use_lighting = p_batch->use_lighting;
	pipeline_key.lcd_blend = p_batch->has_blend;

	switch (p_batch->command_type) {
		case Item::Command::TYPE_RECT:
		case Item::Command::TYPE_NINEPATCH: {
			pipeline = _get_pipeline_specialization_or_ubershader(p_shader_data, pipeline_key, push_constant);
			RD::get_singleton()->draw_list_bind_render_pipeline(p_draw_list, pipeline);
			if (p_batch->has_blend) {
				RD::get_singleton()->draw_list_set_blend_constants(p_draw_list, p_batch->modulate);
			}

			RD::get_singleton()->draw_list_set_push_constant(p_draw_list, &push_constant, sizeof(PushConstant));
			RD::get_singleton()->draw_list_bind_index_array(p_draw_list, shader.quad_index_array);
			RD::get_singleton()->draw_list_draw(p_draw_list, true, p_batch->instance_count);

			if (r_render_info) {
				r_render_info->info[RS::VIEWPORT_RENDER_INFO_TYPE_CANVAS][RS::VIEWPORT_RENDER_INFO_OBJECTS_IN_FRAME] += p_batch->instance_count;
				r_render_info->info[RS::VIEWPORT_RENDER_INFO_TYPE_CANVAS][RS::VIEWPORT_RENDER_INFO_PRIMITIVES_IN_FRAME] += 2 * p_batch->instance_count;
				r_render_info->info[RS::VIEWPORT_RENDER_INFO_TYPE_CANVAS][RS::VIEWPORT_RENDER_INFO_DRAW_CALLS_IN_FRAME]++;
			}
		} break;

		case Item::Command::TYPE_POLYGON: {
			ERR_FAIL_NULL(p_batch->command);

			const Item::CommandPolygon *polygon = static_cast<const Item::CommandPolygon *>(p_batch->command);

			PolygonBuffers *pb = polygon_buffers.polygons.getptr(polygon->polygon.polygon_id);
			ERR_FAIL_NULL(pb);

			pipeline_key.vertex_format_id = pb->vertex_format_id;
			pipeline = _get_pipeline_specialization_or_ubershader(p_shader_data, pipeline_key, push_constant);
			RD::get_singleton()->draw_list_bind_render_pipeline(p_draw_list, pipeline);

			RD::get_singleton()->draw_list_set_push_constant(p_draw_list, &push_constant, sizeof(PushConstant));
			RD::get_singleton()->draw_list_bind_vertex_array(p_draw_list, pb->vertex_array);
			if (pb->indices.is_valid()) {
				RD::get_singleton()->draw_list_bind_index_array(p_draw_list, pb->indices);
			}

			RD::get_singleton()->draw_list_draw(p_draw_list, pb->indices.is_valid());
			if (r_render_info) {
				r_render_info->info[RS::VIEWPORT_RENDER_INFO_TYPE_CANVAS][RS::VIEWPORT_RENDER_INFO_OBJECTS_IN_FRAME]++;
				r_render_info->info[RS::VIEWPORT_RENDER_INFO_TYPE_CANVAS][RS::VIEWPORT_RENDER_INFO_PRIMITIVES_IN_FRAME] += _indices_to_primitives(polygon->primitive, pb->primitive_count);
				r_render_info->info[RS::VIEWPORT_RENDER_INFO_TYPE_CANVAS][RS::VIEWPORT_RENDER_INFO_DRAW_CALLS_IN_FRAME]++;
			}
		} break;

		case Item::Command::TYPE_PRIMITIVE: {
			ERR_FAIL_NULL(p_batch->command);

			const Item::CommandPrimitive *primitive = static_cast<const Item::CommandPrimitive *>(p_batch->command);

			pipeline = _get_pipeline_specialization_or_ubershader(p_shader_data, pipeline_key, push_constant);
			RD::get_singleton()->draw_list_bind_render_pipeline(p_draw_list, pipeline);

			RD::get_singleton()->draw_list_set_push_constant(p_draw_list, &push_constant, sizeof(PushConstant));
			RD::get_singleton()->draw_list_bind_index_array(p_draw_list, primitive_arrays.index_array[MIN(3u, primitive->point_count) - 1]);
			uint32_t instance_count = p_batch->instance_count;
			RD::get_singleton()->draw_list_draw(p_draw_list, true, instance_count);

			if (r_render_info) {
				const RenderingServer::PrimitiveType rs_primitive[5] = { RS::PRIMITIVE_POINTS, RS::PRIMITIVE_POINTS, RS::PRIMITIVE_LINES, RS::PRIMITIVE_TRIANGLES, RS::PRIMITIVE_TRIANGLES };
				r_render_info->info[RS::VIEWPORT_RENDER_INFO_TYPE_CANVAS][RS::VIEWPORT_RENDER_INFO_OBJECTS_IN_FRAME] += instance_count;
				r_render_info->info[RS::VIEWPORT_RENDER_INFO_TYPE_CANVAS][RS::VIEWPORT_RENDER_INFO_PRIMITIVES_IN_FRAME] += _indices_to_primitives(rs_primitive[p_batch->primitive_points], p_batch->primitive_points) * instance_count;
				r_render_info->info[RS::VIEWPORT_RENDER_INFO_TYPE_CANVAS][RS::VIEWPORT_RENDER_INFO_DRAW_CALLS_IN_FRAME]++;
			}
		} break;

		case Item::Command::TYPE_MESH:
		case Item::Command::TYPE_MULTIMESH:
		case Item::Command::TYPE_PARTICLES: {
			ERR_FAIL_NULL(p_batch->command);

			RendererRD::MeshStorage *mesh_storage = RendererRD::MeshStorage::get_singleton();
			RendererRD::ParticlesStorage *particles_storage = RendererRD::ParticlesStorage::get_singleton();

			RID mesh;
			RID mesh_instance;

			if (p_batch->command_type == Item::Command::TYPE_MESH) {
				const Item::CommandMesh *m = static_cast<const Item::CommandMesh *>(p_batch->command);
				mesh = m->mesh;
				mesh_instance = m->mesh_instance;
			} else if (p_batch->command_type == Item::Command::TYPE_MULTIMESH) {
				const Item::CommandMultiMesh *mm = static_cast<const Item::CommandMultiMesh *>(p_batch->command);
				RID multimesh = mm->multimesh;
				mesh = mesh_storage->multimesh_get_mesh(multimesh);

				RID uniform_set = mesh_storage->multimesh_get_2d_uniform_set(multimesh, shader.default_version_rd_shader, TRANSFORMS_UNIFORM_SET);
				RD::get_singleton()->draw_list_bind_uniform_set(p_draw_list, uniform_set, TRANSFORMS_UNIFORM_SET);
			} else if (p_batch->command_type == Item::Command::TYPE_PARTICLES) {
				const Item::CommandParticles *pt = static_cast<const Item::CommandParticles *>(p_batch->command);
				RID particles = pt->particles;
				mesh = particles_storage->particles_get_draw_pass_mesh(particles, 0);

				ERR_BREAK(particles_storage->particles_get_mode(particles) != RS::PARTICLES_MODE_2D);
				particles_storage->particles_request_process(particles);

				if (particles_storage->particles_is_inactive(particles)) {
					break;
				}

				RenderingServerDefault::redraw_request(); // Active particles means redraw request.

				int dpc = particles_storage->particles_get_draw_passes(particles);
				if (dpc == 0) {
					break; // Nothing to draw.
				}

				RID uniform_set = particles_storage->particles_get_instance_buffer_uniform_set(pt->particles, shader.default_version_rd_shader, TRANSFORMS_UNIFORM_SET);
				RD::get_singleton()->draw_list_bind_uniform_set(p_draw_list, uniform_set, TRANSFORMS_UNIFORM_SET);
			}

			if (mesh.is_null()) {
				break;
			}

			uint32_t surf_count = mesh_storage->mesh_get_surface_count(mesh);

			for (uint32_t j = 0; j < surf_count; j++) {
				void *surface = mesh_storage->mesh_get_surface(mesh, j);

				RS::PrimitiveType primitive = mesh_storage->mesh_surface_get_primitive(surface);
				ERR_CONTINUE(primitive < 0 || primitive >= RS::PRIMITIVE_MAX);

				RID vertex_array;
				pipeline_key.variant = primitive == RS::PRIMITIVE_POINTS ? SHADER_VARIANT_ATTRIBUTES_POINTS : SHADER_VARIANT_ATTRIBUTES;
				pipeline_key.render_primitive = _primitive_type_to_render_primitive(primitive);
				pipeline_key.vertex_format_id = RD::INVALID_FORMAT_ID;

				pipeline = _get_pipeline_specialization_or_ubershader(p_shader_data, pipeline_key, push_constant, mesh_instance, surface, j, &vertex_array);
				RD::get_singleton()->draw_list_bind_render_pipeline(p_draw_list, pipeline);

				RD::get_singleton()->draw_list_set_push_constant(p_draw_list, &push_constant, sizeof(PushConstant));

				RID index_array = mesh_storage->mesh_surface_get_index_array(surface, 0);

				if (index_array.is_valid()) {
					RD::get_singleton()->draw_list_bind_index_array(p_draw_list, index_array);
				}

				RD::get_singleton()->draw_list_bind_vertex_array(p_draw_list, vertex_array);
				RD::get_singleton()->draw_list_draw(p_draw_list, index_array.is_valid(), p_batch->mesh_instance_count);

				if (r_render_info) {
					r_render_info->info[RS::VIEWPORT_RENDER_INFO_TYPE_CANVAS][RS::VIEWPORT_RENDER_INFO_OBJECTS_IN_FRAME]++;
					r_render_info->info[RS::VIEWPORT_RENDER_INFO_TYPE_CANVAS][RS::VIEWPORT_RENDER_INFO_PRIMITIVES_IN_FRAME] += _indices_to_primitives(primitive, mesh_storage->mesh_surface_get_vertices_drawn_count(surface)) * p_batch->mesh_instance_count;
					r_render_info->info[RS::VIEWPORT_RENDER_INFO_TYPE_CANVAS][RS::VIEWPORT_RENDER_INFO_DRAW_CALLS_IN_FRAME]++;
				}
			}
		} break;
		case Item::Command::TYPE_TRANSFORM:
		case Item::Command::TYPE_CLIP_IGNORE:
		case Item::Command::TYPE_ANIMATION_SLICE: {
			// Can ignore these as they only impact batch creation.
		} break;
	}
}
