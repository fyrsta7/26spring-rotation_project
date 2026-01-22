	x ^= x >> 17;
	x ^= x << 5;
	*state = x;
	return x;
}

Vector3 VoxelLightBaker::_compute_ray_trace_at_pos(const Vector3 &p_pos, const Vector3 &p_normal, uint32_t *rng_state) {

	int samples_per_quality[3] = { 48, 128, 512 };

	int samples = samples_per_quality[bake_quality];

	//create a basis in Z
	Vector3 v0 = Math::abs(p_normal.z) < 0.999 ? Vector3(0, 0, 1) : Vector3(0, 1, 0);
	Vector3 tangent = v0.cross(p_normal).normalized();
	Vector3 bitangent = tangent.cross(p_normal).normalized();
	Basis normal_xform = Basis(tangent, bitangent, p_normal).transposed();

	float bias = 1.5;
	int max_level = cell_subdiv - 1;
	int size = 1 << max_level;

	Vector3 accum;
	float spread = Math::deg2rad(80.0);

	const Light *light = bake_light.ptr();
	const Cell *cells = bake_cells.ptr();

	// Prevent false sharing when running on OpenMP
	uint32_t local_rng_state = *rng_state;

	for (int i = 0; i < samples; i++) {

		float random_angle1 = (((xorshift32(&local_rng_state) % 65535) / 65535.0) * 2.0 - 1.0) * spread;
		Vector3 axis(0, sin(random_angle1), cos(random_angle1));
		float random_angle2 = ((xorshift32(&local_rng_state) % 65535) / 65535.0) * Math_PI * 2.0;
		Basis rot(Vector3(0, 0, 1), random_angle2);
		axis = rot.xform(axis);

		Vector3 direction = normal_xform.xform(axis).normalized();

		Vector3 advance = direction * _get_normal_advance(direction);

		Vector3 pos = p_pos /*+ Vector3(0.5, 0.5, 0.5)*/ + advance * bias;

		uint32_t cell = CHILD_EMPTY;

		while (cell == CHILD_EMPTY) {

			int x = int(pos.x);
			int y = int(pos.y);
			int z = int(pos.z);

			int ofs_x = 0;
			int ofs_y = 0;
			int ofs_z = 0;
			int half = size / 2;

			if (x < 0 || x >= size)
				break;
			if (y < 0 || y >= size)
				break;
			if (z < 0 || z >= size)
				break;

			//int level_limit = max_level;

			cell = 0; //start from root
			for (int i = 0; i < max_level; i++) {

				const Cell *bc = &cells[cell];

				int child = 0;
				if (x >= ofs_x + half) {
					child |= 1;
					ofs_x += half;
				}
				if (y >= ofs_y + half) {
					child |= 2;
					ofs_y += half;
				}
				if (z >= ofs_z + half) {
					child |= 4;
					ofs_z += half;
				}

				cell = bc->childs[child];
				if (unlikely(cell == CHILD_EMPTY))
					break;

				half >>= 1;
			}

			pos += advance;
		}

		if (unlikely(cell != CHILD_EMPTY)) {
			for (int i = 0; i < 6; i++) {
				//anisotropic read light
				float amount = direction.dot(aniso_normal[i]);
				if (amount <= 0)
					continue;
				accum.x += light[cell].accum[i][0] * amount;
				accum.y += light[cell].accum[i][1] * amount;
				accum.z += light[cell].accum[i][2] * amount;
			}
			accum.x += cells[cell].emission[0];
			accum.y += cells[cell].emission[1];
			accum.z += cells[cell].emission[2];
		}
