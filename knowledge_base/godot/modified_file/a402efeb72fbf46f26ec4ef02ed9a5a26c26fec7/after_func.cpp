		}
	}

	return accum / samples;
}

Error VoxelLightBaker::make_lightmap(const Transform &p_xform, Ref<Mesh> &p_mesh, LightMapData &r_lightmap, bool (*p_bake_time_func)(void *, float, float), void *p_bake_time_ud) {

	//transfer light information to a lightmap
	Ref<Mesh> mesh = p_mesh;

	int width = mesh->get_lightmap_size_hint().x;
	int height = mesh->get_lightmap_size_hint().y;

	//step 1 - create lightmap
	Vector<LightMap> lightmap;
	lightmap.resize(width * height);

	Transform xform = to_cell_space * p_xform;

	//step 2 plot faces to lightmap
	for (int i = 0; i < mesh->get_surface_count(); i++) {
		Array arrays = mesh->surface_get_arrays(i);
		PoolVector<Vector3> vertices = arrays[Mesh::ARRAY_VERTEX];
		PoolVector<Vector3> normals = arrays[Mesh::ARRAY_NORMAL];
		PoolVector<Vector2> uv2 = arrays[Mesh::ARRAY_TEX_UV2];
		PoolVector<int> indices = arrays[Mesh::ARRAY_INDEX];

		ERR_FAIL_COND_V(vertices.size() == 0, ERR_INVALID_PARAMETER);
		ERR_FAIL_COND_V(normals.size() == 0, ERR_INVALID_PARAMETER);
		ERR_FAIL_COND_V(uv2.size() == 0, ERR_INVALID_PARAMETER);

		int vc = vertices.size();
		PoolVector<Vector3>::Read vr = vertices.read();
		PoolVector<Vector3>::Read nr = normals.read();
		PoolVector<Vector2>::Read u2r = uv2.read();
		PoolVector<int>::Read ir;
		int ic = 0;

		if (indices.size()) {
			ic = indices.size();
			ir = indices.read();
		}

		int faces = ic ? ic / 3 : vc / 3;
		for (int i = 0; i < faces; i++) {
			Vector3 vertex[3];
			Vector3 normal[3];
			Vector2 uv[3];
			for (int j = 0; j < 3; j++) {
				int idx = ic ? ir[i * 3 + j] : i * 3 + j;
				vertex[j] = xform.xform(vr[idx]);
				normal[j] = xform.basis.xform(nr[idx]).normalized();
				uv[j] = u2r[idx];
			}

			_plot_triangle(uv, vertex, normal, lightmap.ptrw(), width, height);
		}
	}
	//step 3 perform voxel cone trace on lightmap pixels

	{
		LightMap *lightmap_ptr = lightmap.ptrw();
		uint64_t begin_time = OS::get_singleton()->get_ticks_usec();
		volatile int lines = 0;

		for (int i = 0; i < height; i++) {

		//print_line("bake line " + itos(i) + " / " + itos(height));
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
			for (int j = 0; j < width; j++) {

				//if (i == 125 && j == 280) {

				LightMap *pixel = &lightmap_ptr[i * width + j];
				if (pixel->pos == Vector3())
					continue; //unused, skipe

				//print_line("pos: " + pixel->pos + " normal " + pixel->normal);
				switch (bake_mode) {
					case BAKE_MODE_CONE_TRACE: {
						pixel->light = _compute_pixel_light_at_pos(pixel->pos, pixel->normal) * energy;
					} break;
					case BAKE_MODE_RAY_TRACE: {
						pixel->light = _compute_ray_trace_at_pos(pixel->pos, pixel->normal) * energy;
					} break;
						//	pixel->light = Vector3(1, 1, 1);
						//}
				}
			}

			lines = MAX(lines, i); //for multithread
			if (p_bake_time_func) {
				uint64_t elapsed = OS::get_singleton()->get_ticks_usec() - begin_time;
				float elapsed_sec = double(elapsed) / 1000000.0;
				float remaining = lines < 1 ? 0 : (elapsed_sec / lines) * (height - lines - 1);
				if (p_bake_time_func(p_bake_time_ud, remaining, lines / float(height))) {
					return ERR_SKIP;
				}
			}
		}

		if (bake_mode == BAKE_MODE_RAY_TRACE) {
			//blur
			print_line("bluring, use pos for separatable copy");
			//gauss kernel, 7 step sigma 2
			static const float gauss_kernel[4] = { 0.214607, 0.189879, 0.131514, 0.071303 };
			//horizontal pass
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					if (lightmap_ptr[i * width + j].normal == Vector3()) {
						continue; //empty
					}
					float gauss_sum = gauss_kernel[0];
					Vector3 accum = lightmap_ptr[i * width + j].light * gauss_kernel[0];
					for (int k = 1; k < 4; k++) {
						int new_x = j + k;
						if (new_x >= width || lightmap_ptr[i * width + new_x].normal == Vector3())
							break;
						gauss_sum += gauss_kernel[k];
						accum += lightmap_ptr[i * width + new_x].light * gauss_kernel[k];
					}
					for (int k = 1; k < 4; k++) {
						int new_x = j - k;
						if (new_x < 0 || lightmap_ptr[i * width + new_x].normal == Vector3())
							break;
						gauss_sum += gauss_kernel[k];
						accum += lightmap_ptr[i * width + new_x].light * gauss_kernel[k];
					}

					lightmap_ptr[i * width + j].pos = accum /= gauss_sum;
				}
			}
			//vertical pass
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					if (lightmap_ptr[i * width + j].normal == Vector3())
						continue; //empty, dont write over it anyway
					float gauss_sum = gauss_kernel[0];
					Vector3 accum = lightmap_ptr[i * width + j].pos * gauss_kernel[0];
					for (int k = 1; k < 4; k++) {
						int new_y = i + k;
						if (new_y >= height || lightmap_ptr[new_y * width + j].normal == Vector3())
							break;
						gauss_sum += gauss_kernel[k];
						accum += lightmap_ptr[new_y * width + j].pos * gauss_kernel[k];
					}
					for (int k = 1; k < 4; k++) {
						int new_y = i - k;
						if (new_y < 0 || lightmap_ptr[new_y * width + j].normal == Vector3())
							break;
						gauss_sum += gauss_kernel[k];
						accum += lightmap_ptr[new_y * width + j].pos * gauss_kernel[k];
					}

					lightmap_ptr[i * width + j].light = accum /= gauss_sum;
				}
			}
		}

		//add directional light (do this after blur)
		{
			LightMap *lightmap_ptr = lightmap.ptrw();
			const Cell *cells = bake_cells.ptr();
			const Light *light = bake_light.ptr();
#ifdef _OPENMP
#pragma omp parallel
#endif
			for (int i = 0; i < height; i++) {

			//print_line("bake line " + itos(i) + " / " + itos(height));
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
				for (int j = 0; j < width; j++) {

					//if (i == 125 && j == 280) {

					LightMap *pixel = &lightmap_ptr[i * width + j];
					if (pixel->pos == Vector3())
						continue; //unused, skipe

					int x = int(pixel->pos.x) - 1;
					int y = int(pixel->pos.y) - 1;
					int z = int(pixel->pos.z) - 1;
					Color accum;
					int size = 1 << (cell_subdiv - 1);

					int found = 0;

					for (int k = 0; k < 8; k++) {

						int ofs_x = x;
						int ofs_y = y;
						int ofs_z = z;

						if (k & 1)
							ofs_x++;
						if (k & 2)
							ofs_y++;
						if (k & 4)
							ofs_z++;

						if (x < 0 || x >= size)
							continue;
						if (y < 0 || y >= size)
							continue;
						if (z < 0 || z >= size)
							continue;

						uint32_t cell = _find_cell_at_pos(cells, ofs_x, ofs_y, ofs_z);

						if (cell == CHILD_EMPTY)
							continue;
						for (int l = 0; l < 6; l++) {
							float s = pixel->normal.dot(aniso_normal[l]);
							if (s < 0)
								s = 0;
							accum.r += light[cell].direct_accum[l][0] * s;
							accum.g += light[cell].direct_accum[l][1] * s;
							accum.b += light[cell].direct_accum[l][2] * s;
						}
						found++;
					}
					if (found) {
						accum /= found;
						pixel->light.x += accum.r;
						pixel->light.y += accum.g;
						pixel->light.z += accum.b;
					}
				}
			}
		}

		{
			//fill gaps with neighbour vertices to avoid filter fades to black on edges

			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					if (lightmap_ptr[i * width + j].normal != Vector3()) {
						continue; //filled, skip
					}

					//this can't be made separatable..

					int closest_i = -1, closest_j = 1;
					float closest_dist = 1e20;

					const int margin = 3;
					for (int y = i - margin; y <= i + margin; y++) {
						for (int x = j - margin; x <= j + margin; x++) {

							if (x == j && y == i)
								continue;
							if (x < 0 || x >= width)
								continue;
							if (y < 0 || y >= height)
								continue;
							if (lightmap_ptr[y * width + x].normal == Vector3())
								continue; //also ensures that blitted stuff is not reused

							float dist = Vector2(i - y, j - x).length();
							if (dist > closest_dist)
								continue;

							closest_dist = dist;
							closest_i = y;
							closest_j = x;
						}
					}

					if (closest_i != -1) {
						lightmap_ptr[i * width + j].light = lightmap_ptr[closest_i * width + closest_j].light;
					}
				}
			}
		}

		{
			//fill the lightmap data
			r_lightmap.width = width;
			r_lightmap.height = height;
			r_lightmap.light.resize(lightmap.size() * 3);
			PoolVector<float>::Write w = r_lightmap.light.write();
			for (int i = 0; i < lightmap.size(); i++) {
				w[i * 3 + 0] = lightmap[i].light.x;
				w[i * 3 + 1] = lightmap[i].light.y;
				w[i * 3 + 2] = lightmap[i].light.z;
			}
		}

#if 0
		{
			PoolVector<uint8_t> img;
			int ls = lightmap.size();
			img.resize(ls * 3);
			{
				PoolVector<uint8_t>::Write w = img.write();
				for (int i = 0; i < ls; i++) {
					w[i * 3 + 0] = CLAMP(lightmap_ptr[i].light.x * 255, 0, 255);
					w[i * 3 + 1] = CLAMP(lightmap_ptr[i].light.y * 255, 0, 255);
					w[i * 3 + 2] = CLAMP(lightmap_ptr[i].light.z * 255, 0, 255);
					//w[i * 3 + 0] = CLAMP(lightmap_ptr[i].normal.x * 255, 0, 255);
					//w[i * 3 + 1] = CLAMP(lightmap_ptr[i].normal.y * 255, 0, 255);
					//w[i * 3 + 2] = CLAMP(lightmap_ptr[i].normal.z * 255, 0, 255);
					//w[i * 3 + 0] = CLAMP(lightmap_ptr[i].pos.x / (1 << (cell_subdiv - 1)) * 255, 0, 255);
					//w[i * 3 + 1] = CLAMP(lightmap_ptr[i].pos.y / (1 << (cell_subdiv - 1)) * 255, 0, 255);
					//w[i * 3 + 2] = CLAMP(lightmap_ptr[i].pos.z / (1 << (cell_subdiv - 1)) * 255, 0, 255);
				}
			}

			Ref<Image> image;
			image.instance();
			image->create(width, height, false, Image::FORMAT_RGB8, img);

			String name = p_mesh->get_name();
			if (name == "") {
				name = "Mesh" + itos(p_mesh->get_instance_id());
			}
			image->save_png(name + ".png");
