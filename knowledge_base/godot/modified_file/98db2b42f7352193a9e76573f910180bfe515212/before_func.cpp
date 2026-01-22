void ParticleProcessMaterial::_update_shader() {
	dirty_materials->remove(&element);

	MaterialKey mk = _compute_key();
	if (mk == current_key) {
		return; //no update required in the end
	}

	if (shader_map.has(current_key)) {
		shader_map[current_key].users--;
		if (shader_map[current_key].users == 0) {
			//deallocate shader, as it's no longer in use
			RS::get_singleton()->free(shader_map[current_key].shader);
			shader_map.erase(current_key);
		}
	}

	current_key = mk;

	if (shader_map.has(mk)) {
		RS::get_singleton()->material_set_shader(_get_material(), shader_map[mk].shader);
		shader_map[mk].users++;
		return;
	}
	//must create a shader!

	// Add a comment to describe the shader origin (useful when converting to ShaderMaterial).
	String code = "// NOTE: Shader automatically converted from " VERSION_NAME " " VERSION_FULL_CONFIG "'s ParticleProcessMaterial.\n\n";

	code += "shader_type particles;\n";
	code += "render_mode disable_velocity;\n";

	if (collision_scale) {
		code += "render_mode collision_use_scale;\n";
	}

	code += "uniform vec3 direction;\n";
	code += "uniform float spread;\n";
	code += "uniform float flatness;\n";

	code += "uniform float inherit_emitter_velocity_ratio = 0;\n";

	code += "uniform float initial_linear_velocity_min;\n";
	code += "uniform float initial_linear_velocity_max;\n";

	code += "uniform float directional_velocity_min;\n";
	code += "uniform float directional_velocity_max;\n";

	code += "uniform float angular_velocity_min;\n";
	code += "uniform float angular_velocity_max;\n";

	code += "uniform float orbit_velocity_min;\n";
	code += "uniform float orbit_velocity_max;\n";

	code += "uniform float radial_velocity_min;\n";
	code += "uniform float radial_velocity_max;\n";

	code += "uniform float linear_accel_min;\n";
	code += "uniform float linear_accel_max;\n";

	code += "uniform float radial_accel_min;\n";
	code += "uniform float radial_accel_max;\n";

	code += "uniform float tangent_accel_min;\n";
	code += "uniform float tangent_accel_max;\n";

	code += "uniform float damping_min;\n";
	code += "uniform float damping_max;\n";

	code += "uniform float initial_angle_min;\n";
	code += "uniform float initial_angle_max;\n";

	code += "uniform float scale_min;\n";
	code += "uniform float scale_max;\n";

	code += "uniform float hue_variation_min;\n";
	code += "uniform float hue_variation_max;\n";

	code += "uniform float anim_speed_min;\n";
	code += "uniform float anim_speed_max;\n";

	code += "uniform float anim_offset_min;\n";
	code += "uniform float anim_offset_max;\n";

	code += "uniform float lifetime_randomness;\n";
	code += "uniform vec3 emission_shape_offset = vec3(0.);\n";
	code += "uniform vec3 emission_shape_scale = vec3(1.);\n";

	code += "uniform vec3 velocity_pivot = vec3(0.);\n";

	if (tex_parameters[PARAM_SCALE_OVER_VELOCITY].is_valid()) {
		code += "uniform float scale_over_velocity_min = 0.0;\n";
		code += "uniform float scale_over_velocity_max = 5.0;\n";
	}

	switch (emission_shape) {
		case EMISSION_SHAPE_POINT: {
			//do none
		} break;
		case EMISSION_SHAPE_SPHERE: {
			code += "uniform float emission_sphere_radius;\n";
		} break;
		case EMISSION_SHAPE_SPHERE_SURFACE: {
			code += "uniform float emission_sphere_radius;\n";
		} break;
		case EMISSION_SHAPE_BOX: {
			code += "uniform vec3 emission_box_extents;\n";
		} break;
		case EMISSION_SHAPE_DIRECTED_POINTS: {
			code += "uniform sampler2D emission_texture_normal : hint_default_black;\n";
			[[fallthrough]];
		}
		case EMISSION_SHAPE_POINTS: {
			code += "uniform sampler2D emission_texture_points : hint_default_black;\n";
			code += "uniform int emission_texture_point_count;\n";
			if (emission_color_texture.is_valid()) {
				code += "uniform sampler2D emission_texture_color : hint_default_white;\n";
			}
		} break;
		case EMISSION_SHAPE_RING: {
			code += "uniform vec3 " + shader_names->emission_ring_axis + ";\n";
			code += "uniform float " + shader_names->emission_ring_height + ";\n";
			code += "uniform float " + shader_names->emission_ring_radius + ";\n";
			code += "uniform float " + shader_names->emission_ring_inner_radius + ";\n";
		} break;
		case EMISSION_SHAPE_MAX: { // Max value for validity check.
			break;
		}
	}

	if (sub_emitter_mode != SUB_EMITTER_DISABLED && !RenderingServer::get_singleton()->is_low_end()) {
		if (sub_emitter_mode == SUB_EMITTER_CONSTANT) {
			code += "uniform float sub_emitter_frequency;\n";
		}
		if (sub_emitter_mode == SUB_EMITTER_AT_END) {
			code += "uniform int sub_emitter_amount_at_end;\n";
		}
		if (sub_emitter_mode == SUB_EMITTER_AT_COLLISION) {
			code += "uniform int sub_emitter_amount_at_collision;\n";
		}
		code += "uniform bool sub_emitter_keep_velocity;\n";
	}

	code += "uniform vec4 color_value : source_color;\n";

	code += "uniform vec3 gravity;\n";

	if (color_ramp.is_valid()) {
		code += "uniform sampler2D color_ramp : repeat_disable;\n";
	}

	if (color_initial_ramp.is_valid()) {
		code += "uniform sampler2D color_initial_ramp : repeat_disable;\n";
	}
	if (alpha_curve.is_valid()) {
		code += "uniform sampler2D alpha_curve : repeat_disable;\n";
	}
	if (emission_curve.is_valid()) {
		code += "uniform sampler2D emission_curve : repeat_disable;\n";
	}

	if (tex_parameters[PARAM_INITIAL_LINEAR_VELOCITY].is_valid()) {
		code += "uniform sampler2D linear_velocity_texture : repeat_disable;\n";
	}
	if (tex_parameters[PARAM_ORBIT_VELOCITY].is_valid()) {
		code += "uniform sampler2D orbit_velocity_curve : repeat_disable;\n";
	}
	if (tex_parameters[PARAM_ANGULAR_VELOCITY].is_valid()) {
		code += "uniform sampler2D angular_velocity_texture : repeat_disable;\n";
	}
	if (tex_parameters[PARAM_LINEAR_ACCEL].is_valid()) {
		code += "uniform sampler2D linear_accel_texture : repeat_disable;\n";
	}
	if (tex_parameters[PARAM_RADIAL_ACCEL].is_valid()) {
		code += "uniform sampler2D radial_accel_texture : repeat_disable;\n";
	}
	if (tex_parameters[PARAM_TANGENTIAL_ACCEL].is_valid()) {
		code += "uniform sampler2D tangent_accel_texture : repeat_disable;\n";
	}
	if (tex_parameters[PARAM_DAMPING].is_valid()) {
		code += "uniform sampler2D damping_texture : repeat_disable;\n";
	}
	if (tex_parameters[PARAM_ANGLE].is_valid()) {
		code += "uniform sampler2D angle_texture : repeat_disable;\n";
	}
	if (tex_parameters[PARAM_SCALE].is_valid()) {
		code += "uniform sampler2D scale_curve : repeat_disable;\n";
	}
	if (tex_parameters[PARAM_HUE_VARIATION].is_valid()) {
		code += "uniform sampler2D hue_rot_curve : repeat_disable;\n";
	}
	if (tex_parameters[PARAM_ANIM_SPEED].is_valid()) {
		code += "uniform sampler2D animation_speed_curve : repeat_disable;\n";
	}
	if (tex_parameters[PARAM_ANIM_OFFSET].is_valid()) {
		code += "uniform sampler2D animation_offset_curve : repeat_disable;\n";
	}
	if (tex_parameters[PARAM_RADIAL_VELOCITY].is_valid()) {
		code += "uniform sampler2D radial_velocity_curve : repeat_disable;\n";
	}
	if (tex_parameters[PARAM_SCALE_OVER_VELOCITY].is_valid()) {
		code += "uniform sampler2D scale_over_velocity_curve : repeat_disable;\n";
	}
	if (tex_parameters[PARAM_DIRECTIONAL_VELOCITY].is_valid()) {
		code += "uniform sampler2D directional_velocity_curve: repeat_disable;\n";
	}
	if (velocity_limit_curve.is_valid()) {
		code += "uniform sampler2D velocity_limit_curve: repeat_disable;\n";
	}

	if (collision_mode == COLLISION_RIGID) {
		code += "uniform float collision_friction;\n";
		code += "uniform float collision_bounce;\n";
	}

	if (turbulence_enabled) {
		code += "uniform float turbulence_noise_strength;\n";
		code += "uniform float turbulence_noise_scale;\n";
		code += "uniform float turbulence_influence_min;\n";
		code += "uniform float turbulence_influence_max;\n";
		code += "uniform float turbulence_initial_displacement_min;\n";
		code += "uniform float turbulence_initial_displacement_max;\n";
		code += "uniform float turbulence_noise_speed_random;\n";
		code += "uniform vec3 turbulence_noise_speed = vec3(1.0, 1.0, 1.0);\n";
		if (tex_parameters[PARAM_TURB_INFLUENCE_OVER_LIFE].is_valid()) {
			code += "uniform sampler2D turbulence_influence_over_life;\n";
		}
		if (turbulence_color_ramp.is_valid()) {
			code += "uniform sampler2D turbulence_color_ramp;\n";
		}
		code += "\n";

		//functions for 3D noise / turbulence
		code += "\n\n";
		code += "vec4 grad(vec4 p) {\n";
		code += "	p = fract(vec4(\n";
		code += "		dot(p, vec4(0.143081, 0.001724, 0.280166, 0.262771)),\n";
		code += "		dot(p, vec4(0.645401, -0.047791, -0.146698, 0.595016)),\n";
		code += "		dot(p, vec4(-0.499665, -0.095734, 0.425674, -0.207367)),\n";
		code += "		dot(p, vec4(-0.013596, -0.848588, 0.423736, 0.17044))));\n";
		code += "	return fract((p.xyzw * p.yzwx) * 2365.952041) * 2.0 - 1.0;\n";
		code += "}\n";
		code += "float noise(vec4 coord) {\n";
		code += "	// Domain rotation to improve the look of XYZ slices + animation patterns.\n";
		code += "	coord = vec4(\n";
		code += "		coord.xyz + dot(coord, vec4(vec3(-0.1666667), -0.5)),\n";
		code += "		dot(coord, vec4(0.5)));\n\n";
		code += "	vec4 base = floor(coord), delta = coord - base;\n\n";
		code += "	vec4 grad_0000 = grad(base + vec4(0.0, 0.0, 0.0, 0.0)), grad_1000 = grad(base + vec4(1.0, 0.0, 0.0, 0.0));\n";
		code += "	vec4 grad_0100 = grad(base + vec4(0.0, 1.0, 0.0, 0.0)), grad_1100 = grad(base + vec4(1.0, 1.0, 0.0, 0.0));\n";
		code += "	vec4 grad_0010 = grad(base + vec4(0.0, 0.0, 1.0, 0.0)), grad_1010 = grad(base + vec4(1.0, 0.0, 1.0, 0.0));\n";
		code += "	vec4 grad_0110 = grad(base + vec4(0.0, 1.0, 1.0, 0.0)), grad_1110 = grad(base + vec4(1.0, 1.0, 1.0, 0.0));\n";
		code += "	vec4 grad_0001 = grad(base + vec4(0.0, 0.0, 0.0, 1.0)), grad_1001 = grad(base + vec4(1.0, 0.0, 0.0, 1.0));\n";
		code += "	vec4 grad_0101 = grad(base + vec4(0.0, 1.0, 0.0, 1.0)), grad_1101 = grad(base + vec4(1.0, 1.0, 0.0, 1.0));\n";
		code += "	vec4 grad_0011 = grad(base + vec4(0.0, 0.0, 1.0, 1.0)), grad_1011 = grad(base + vec4(1.0, 0.0, 1.0, 1.0));\n";
		code += "	vec4 grad_0111 = grad(base + vec4(0.0, 1.0, 1.0, 1.0)), grad_1111 = grad(base + vec4(1.0, 1.0, 1.0, 1.0));\n\n";
		code += "	vec4 result_0123 = vec4(\n";
		code += "		dot(delta - vec4(0.0, 0.0, 0.0, 0.0), grad_0000), dot(delta - vec4(1.0, 0.0, 0.0, 0.0), grad_1000),\n";
		code += "		dot(delta - vec4(0.0, 1.0, 0.0, 0.0), grad_0100), dot(delta - vec4(1.0, 1.0, 0.0, 0.0), grad_1100));\n";
		code += "	vec4 result_4567 = vec4(\n";
		code += "		dot(delta - vec4(0.0, 0.0, 1.0, 0.0), grad_0010), dot(delta - vec4(1.0, 0.0, 1.0, 0.0), grad_1010),\n";
		code += "		dot(delta - vec4(0.0, 1.0, 1.0, 0.0), grad_0110), dot(delta - vec4(1.0, 1.0, 1.0, 0.0), grad_1110));\n";
		code += "	vec4 result_89AB = vec4(\n";
		code += "		dot(delta - vec4(0.0, 0.0, 0.0, 1.0), grad_0001), dot(delta - vec4(1.0, 0.0, 0.0, 1.0), grad_1001),\n";
		code += "		dot(delta - vec4(0.0, 1.0, 0.0, 1.0), grad_0101), dot(delta - vec4(1.0, 1.0, 0.0, 1.0), grad_1101));\n";
		code += "	vec4 result_CDEF = vec4(\n";
		code += "		dot(delta - vec4(0.0, 0.0, 1.0, 1.0), grad_0011), dot(delta - vec4(1.0, 0.0, 1.0, 1.0), grad_1011),\n";
		code += "		dot(delta - vec4(0.0, 1.0, 1.0, 1.0), grad_0111), dot(delta - vec4(1.0, 1.0, 1.0, 1.0), grad_1111));\n\n";
		code += "	vec4 fade = delta * delta * delta * (10.0 + delta * (-15.0 + delta * 6.0));\n";
		code += "	vec4 result_W0 = mix(result_0123, result_89AB, fade.w), result_W1 = mix(result_4567, result_CDEF, fade.w);\n";
		code += "	vec4 result_WZ = mix(result_W0, result_W1, fade.z);\n";
		code += "	vec2 result_WZY = mix(result_WZ.xy, result_WZ.zw, fade.y);\n";
		code += "	return mix(result_WZY.x, result_WZY.y, fade.x);\n";
		code += "}\n\n";
		code += "// Curl 3D and three-noise function with friendly permission by Isaac Cohen.\n";
		code += "// Modified to accept 4D noise.\n";
		code += "vec3 noise_3x(vec4 p) {\n";
		code += "	float s = noise(p);\n";
		code += "	float s1 = noise(p + vec4(vec3(0.0), 1.7320508 * 2048.333333));\n";
		code += "	float s2 = noise(p - vec4(vec3(0.0), 1.7320508 * 2048.333333));\n";
		code += "	vec3 c = vec3(s, s1, s2);\n";
		code += "	return c;\n";
		code += "}\n";
		code += "vec3 curl_3d(vec4 p, float c) {\n";
		code += "	float epsilon = 0.001 + c;\n";
		code += "	vec4 dx = vec4(epsilon, 0.0, 0.0, 0.0);\n";
		code += "	vec4 dy = vec4(0.0, epsilon, 0.0, 0.0);\n";
		code += "	vec4 dz = vec4(0.0, 0.0, epsilon, 0.0);\n";
		code += "	vec3 x0 = noise_3x(p - dx).xyz;\n";
		code += "	vec3 x1 = noise_3x(p + dx).xyz;\n";
		code += "	vec3 y0 = noise_3x(p - dy).xyz;\n";
		code += "	vec3 y1 = noise_3x(p + dy).xyz;\n";
		code += "	vec3 z0 = noise_3x(p - dz).xyz;\n";
		code += "	vec3 z1 = noise_3x(p + dz).xyz;\n";
		code += "	float x = (y1.z - y0.z) - (z1.y - z0.y);\n";
		code += "	float y = (z1.x - z0.x) - (x1.z - x0.z);\n";
		code += "	float z = (x1.y - x0.y) - (y1.x - y0.x);\n";
		code += "	return normalize(vec3(x, y, z));\n";
		code += "}\n";
		code += "vec3 get_noise_direction(vec3 pos) {\n";
		code += "	float adj_contrast = max((turbulence_noise_strength - 1.0), 0.0) * 70.0;\n";
		code += "	vec4 noise_time = TIME * vec4(turbulence_noise_speed, turbulence_noise_speed_random);\n";
		code += "	vec4 noise_pos = vec4(pos * turbulence_noise_scale, 0.0);\n";
		code += "	vec3 noise_direction = curl_3d(noise_pos + noise_time, adj_contrast);\n";
		code += "	noise_direction = mix(0.9 * noise_direction, noise_direction, turbulence_noise_strength - 9.0);\n";
		code += "	return noise_direction;\n";
		code += "}\n";
	}
	code += "vec4 rotate_hue(vec4 current_color, float hue_rot_angle){\n";
	code += "	float hue_rot_c = cos(hue_rot_angle);\n";
	code += "	float hue_rot_s = sin(hue_rot_angle);\n";
	code += "	mat4 hue_rot_mat = mat4(vec4(0.299, 0.587, 0.114, 0.0),\n";
	code += "			vec4(0.299, 0.587, 0.114, 0.0),\n";
	code += "			vec4(0.299, 0.587, 0.114, 0.0),\n";
	code += "			vec4(0.000, 0.000, 0.000, 1.0)) +\n";
	code += "		mat4(vec4(0.701, -0.587, -0.114, 0.0),\n";
	code += "			vec4(-0.299, 0.413, -0.114, 0.0),\n";
	code += "			vec4(-0.300, -0.588, 0.886, 0.0),\n";
	code += "			vec4(0.000, 0.000, 0.000, 0.0)) * hue_rot_c +\n";
	code += "		mat4(vec4(0.168, 0.330, -0.497, 0.0),\n";
	code += "			vec4(-0.328, 0.035,  0.292, 0.0),\n";
	code += "			vec4(1.250, -1.050, -0.203, 0.0),\n";
	code += "			vec4(0.000, 0.000, 0.000, 0.0)) * hue_rot_s;\n";
	code += "	return hue_rot_mat * current_color;\n";
	code += "}\n";

	//need a random function
	code += "\n\n";
	code += "float rand_from_seed(inout uint seed) {\n";
	code += "	int k;\n";
	code += "	int s = int(seed);\n";
	code += "	if (s == 0)\n";
	code += "	s = 305420679;\n";
	code += "	k = s / 127773;\n";
	code += "	s = 16807 * (s - k * 127773) - 2836 * k;\n";
	code += "	if (s < 0)\n";
	code += "		s += 2147483647;\n";
	code += "	seed = uint(s);\n";
	code += "	return float(seed % uint(65536)) / 65535.0;\n";
	code += "}\n";
	code += "\n";

	code += "float rand_from_seed_m1_p1(inout uint seed) {\n";
	code += "	return rand_from_seed(seed) * 2.0 - 1.0;\n";
	code += "}\n";
	code += "\n";

	//improve seed quality
	code += "uint hash(uint x) {\n";
	code += "	x = ((x >> uint(16)) ^ x) * uint(73244475);\n";
	code += "	x = ((x >> uint(16)) ^ x) * uint(73244475);\n";
	code += "	x = (x >> uint(16)) ^ x;\n";
	code += "	return x;\n";
	code += "}\n";
	code += "\n";

	code += "struct DisplayParameters{\n";
	code += "	vec3 scale;\n";
	code += "	float hue_rotation;\n";
	code += "	float animation_speed;\n";
	code += "	float animation_offset;\n";
	code += "	float lifetime;\n";
	code += "	vec4 color;\n";
	code += "};\n";
	code += "\n";
	code += "struct DynamicsParameters{\n";
	code += "	float angle;\n";
	code += "	float angular_velocity;\n";
	code += "	float initial_velocity_multiplier;\n";
	code += "	float directional_velocity;\n";
	code += "	float radial_velocity;\n";
	code += "	float orbit_velocity;\n";
	if (turbulence_enabled) {
		code += "	float turb_influence;\n";
	}
	code += "};\n";
	code += "struct PhysicalParameters{\n";
	code += "	float linear_accel;\n";
	code += "	float radial_accel;\n";
	code += "	float tangent_accel;\n";
	code += "	float damping;\n";
	code += "};\n";

	code += "\n";
	code += "void calculate_initial_physical_params(inout PhysicalParameters params, inout uint alt_seed){\n";
	code += "	params.linear_accel = mix(linear_accel_min, linear_accel_min ,rand_from_seed(alt_seed));\n";
	code += "	params.radial_accel = mix(radial_accel_min, radial_accel_min,rand_from_seed(alt_seed));\n";
	code += "	params.tangent_accel = mix(tangent_accel_min, tangent_accel_max,rand_from_seed(alt_seed));\n";
	code += "	params.damping = mix(damping_min, damping_max,rand_from_seed(alt_seed));\n";
	code += "}\n";
	code += "\n";
	code += "void calculate_initial_dynamics_params(inout DynamicsParameters params,inout uint alt_seed){\n";
	code += "	// -------------------- DO NOT REORDER OPERATIONS, IT BREAKS VISUAL COMPATIBILITY\n";
	code += "	// -------------------- ADD NEW OPERATIONS AT THE BOTTOM\n";
	code += "	params.angle = mix(initial_angle_min, initial_angle_max, rand_from_seed(alt_seed));\n";
	code += "	params.angular_velocity = mix(angular_velocity_min, angular_velocity_max, rand_from_seed(alt_seed));\n";
	code += "	params.initial_velocity_multiplier = mix(initial_linear_velocity_min, initial_linear_velocity_max,rand_from_seed(alt_seed));\n";
	code += "	params.directional_velocity = mix(directional_velocity_min, directional_velocity_max,rand_from_seed(alt_seed));\n";
	code += "	params.radial_velocity = mix(radial_velocity_min, radial_velocity_max,rand_from_seed(alt_seed));\n";
	code += "	params.orbit_velocity = mix(orbit_velocity_min, orbit_velocity_max,rand_from_seed(alt_seed));\n";
	if (turbulence_enabled) {
		code += "   params.turb_influence = mix(turbulence_influence_min,turbulence_influence_max,rand_from_seed(alt_seed));\n";
	}
	code += "}\n";
	code += "void calculate_initial_display_params(inout DisplayParameters params,inout uint alt_seed){\n";
	code += "	// -------------------- DO NOT REORDER OPERATIONS, IT BREAKS VISUAL COMPATIBILITY\n";
	code += "	// -------------------- ADD NEW OPERATIONS AT THE BOTTOM\n";
	code += "	float pi = 3.14159;\n";
	code += "	float degree_to_rad = pi / 180.0;\n";

	code += "   params.scale = vec3(mix(scale_min, scale_max, rand_from_seed(alt_seed)));\n";
	code += "   params.scale = sign(params.scale) * max(abs(params.scale), 0.001);\n";
	code += "	params.hue_rotation =  pi * 2.0 * mix(hue_variation_min, hue_variation_max, rand_from_seed(alt_seed));\n";
	code += "	params.animation_speed = mix(anim_speed_min, anim_speed_max, rand_from_seed(alt_seed));\n";
	code += "	params.animation_offset = mix(anim_offset_min, anim_offset_max, rand_from_seed(alt_seed));\n";
	code += "	params.lifetime = (1.0 - lifetime_randomness * rand_from_seed(alt_seed));\n";
	code += "	params.color = color_value;\n";
	if (color_initial_ramp.is_valid()) {
		code += "	params.color *= texture(color_initial_ramp, vec2(rand_from_seed(alt_seed)));\n";
	}
	if (emission_color_texture.is_valid() && (emission_shape == EMISSION_SHAPE_POINTS || emission_shape == EMISSION_SHAPE_DIRECTED_POINTS)) {
		code += "	int point = min(emission_texture_point_count - 1, int(rand_from_seed(alt_seed) * float(emission_texture_point_count)));\n";
		code += "	ivec2 emission_tex_size = textureSize(emission_texture_points, 0);\n";
		code += "	ivec2 emission_tex_ofs = ivec2(point % emission_tex_size.x, point / emission_tex_size.x);\n";
		code += "	params.color *= texelFetch(emission_texture_color, emission_tex_ofs, 0);\n";
	}
	code += "}\n";

	// process display parameters that are bound solely by lifetime
	code += "void process_display_param(inout DisplayParameters parameters, float lifetime){\n";
	code += "	// compile-time add textures\n";
	if (tex_parameters[PARAM_SCALE].is_valid()) {
		code += "	parameters.scale *= texture(scale_curve, vec2(lifetime)).rgb;\n";
	}
	if (tex_parameters[PARAM_HUE_VARIATION].is_valid()) {
		code += "	parameters.hue_rotation *= texture(hue_rot_curve, vec2(lifetime)).r;\n";
	}
	if (tex_parameters[PARAM_ANIM_OFFSET].is_valid()) {
		code += "	parameters.animation_offset += texture(animation_offset_curve, vec2(lifetime)).r;\n";
	}
	if (tex_parameters[PARAM_ANIM_SPEED].is_valid()) {
		code += "	parameters.animation_speed *= texture(animation_speed_curve, vec2(lifetime)).r;\n";
	}
	if (color_ramp.is_valid()) {
		code += "   parameters.color *= texture(color_ramp, vec2(lifetime));\n";
	}
	if (alpha_curve.is_valid()) {
		code += "	parameters.color.a *= texture(alpha_curve, vec2(lifetime)).r;\n";
	}
	code += "	parameters.color = rotate_hue(parameters.color, parameters.hue_rotation);\n";
	if (emission_curve.is_valid()) {
		code += "	parameters.color.rgb *= 1.0 + texture(emission_curve, vec2(lifetime)).r;\n";
	}
	code += "}\n";

	code += "vec3 calculate_initial_position(inout uint alt_seed) {\n";
	code += "	float pi = 3.14159;\n";
	code += "	float degree_to_rad = pi / 180.0;\n";
	code += "	vec3 pos = vec3(0.);\n";
	if (emission_shape == EMISSION_SHAPE_POINT) {
		code += "	 pos = vec3(0.);\n";
	}
	if (emission_shape == EMISSION_SHAPE_SPHERE) {
		code += "		float s = rand_from_seed(alt_seed) * 2.0 - 1.0;\n";
		code += "		float t = rand_from_seed(alt_seed) * 2.0 * pi;\n";
		code += "		float p = rand_from_seed(alt_seed);\n";
		code += "		float radius = emission_sphere_radius * sqrt(1.0 - s * s);\n";
		code += "		pos = mix(vec3(0.0, 0.0, 0.0), vec3(radius * cos(t), radius * sin(t), emission_sphere_radius * s), p);\n";
	}

	if (emission_shape == EMISSION_SHAPE_SPHERE_SURFACE) {
		code += "		float s = rand_from_seed(alt_seed) * 2.0 - 1.0;\n";
		code += "		float t = rand_from_seed(alt_seed) * 2.0 * pi;\n";
		code += "		float radius = emission_sphere_radius * sqrt(1.0 - s * s);\n";
		code += "		pos = vec3(radius * cos(t), radius * sin(t), emission_sphere_radius * s);\n";
	}
	if (emission_shape == EMISSION_SHAPE_BOX) {
		code += "		pos = vec3(rand_from_seed(alt_seed) * 2.0 - 1.0, rand_from_seed(alt_seed) * 2.0 - 1.0, rand_from_seed(alt_seed) * 2.0 - 1.0) * emission_box_extents;\n";
	}
	if (emission_shape == EMISSION_SHAPE_POINTS || emission_shape == EMISSION_SHAPE_DIRECTED_POINTS) {
		code += "		int point = min(emission_texture_point_count - 1, int(rand_from_seed(alt_seed) * float(emission_texture_point_count)));\n";
		code += "		ivec2 emission_tex_size = textureSize(emission_texture_points, 0);\n";
		code += "		ivec2 emission_tex_ofs = ivec2(point % emission_tex_size.x, point / emission_tex_size.x);\n";
		code += "		pos = texelFetch(emission_texture_points, emission_tex_ofs, 0).xyz;\n";
	}
	if (emission_shape == EMISSION_SHAPE_RING) {
		code += "		\n";
		code += "		float ring_spawn_angle = rand_from_seed(alt_seed) * 2.0 * pi;\n";
		code += "		float ring_random_radius = rand_from_seed(alt_seed) * (emission_ring_radius - emission_ring_inner_radius) + emission_ring_inner_radius;\n";
		code += "		vec3 axis = normalize(emission_ring_axis);\n";
		code += "		vec3 ortho_axis = vec3(0.0);\n";
		code += "		if (axis == vec3(1.0, 0.0, 0.0)) {\n";
		code += "			ortho_axis = cross(axis, vec3(0.0, 1.0, 0.0));\n";
		code += "		} else {\n";
		code += " 			ortho_axis = cross(axis, vec3(1.0, 0.0, 0.0));\n";
		code += "		}\n";
		code += "		ortho_axis = normalize(ortho_axis);\n";
		code += "		float s = sin(ring_spawn_angle);\n";
		code += "		float c = cos(ring_spawn_angle);\n";
		code += "		float oc = 1.0 - c;\n";
		code += "		ortho_axis = mat3(\n";
		code += "			vec3(c + axis.x * axis.x * oc, axis.x * axis.y * oc - axis.z * s, axis.x * axis.z *oc + axis.y * s),\n";
		code += "			vec3(axis.x * axis.y * oc + s * axis.z, c + axis.y * axis.y * oc, axis.y * axis.z * oc - axis.x * s),\n";
		code += "			vec3(axis.z * axis.x * oc - axis.y * s, axis.z * axis.y * oc + axis.x * s, c + axis.z * axis.z * oc)\n";
		code += "			) * ortho_axis;\n";
		code += "		ortho_axis = normalize(ortho_axis);\n";
		code += "		pos = ortho_axis * ring_random_radius + (rand_from_seed(alt_seed) * emission_ring_height - emission_ring_height / 2.0) * axis;\n";
	}

	code += "	return pos * emission_shape_scale + emission_shape_offset;\n";
	code += "}\n";
	code += "\n";
	if (tex_parameters[PARAM_ORBIT_VELOCITY].is_valid() || particle_flags[PARTICLE_FLAG_DISABLE_Z]) {
		code += "vec3 process_orbit_displacement(DynamicsParameters param, float lifetime, inout uint alt_seed, mat4 transform, mat4 emission_transform,float delta, float total_lifetime){\n";
		// No reason to run all these expensive calculation below if we have no orbit velocity
		// HOWEVER
		// May be a bad idea for fps consistency?
		code += "if(abs(param.orbit_velocity) < 0.01 || delta < 0.001){ return vec3(0.0);}\n";
		code += "\n";
		code += "	vec3 displacement = vec3(0.);\n";
		code += "	float pi = 3.14159;\n";
		code += "	float degree_to_rad = pi / 180.0;\n";
		if (particle_flags[PARTICLE_FLAG_DISABLE_Z]) {
			code += "	float orbit_amount = param.orbit_velocity;\n";

			if (tex_parameters[PARAM_ORBIT_VELOCITY].is_valid()) {
				code += "   orbit_amount *= texture(orbit_velocity_curve, vec2(lifetime)).r;\n";
			}
			code += "	if (orbit_amount != 0.0) {\n";
			code += "       vec3 pos = transform[3].xyz;\n";
			code += "       vec3 org = emission_transform[3].xyz;\n";
			code += "       vec3 diff = pos - org;\n";
			code += "	     float ang = orbit_amount * pi * 2.0;\n";
			code += "	     mat2 rot = mat2(vec2(cos(ang), -sin(ang)), vec2(sin(ang), cos(ang)));\n";
			code += "	     displacement.xy -= diff.xy;\n";
			code += "        displacement.xy += rot * diff.xy;\n";
			code += "	}\n";
		} else {
			code += "	vec3 orbit_velocities = vec3(param.orbit_velocity);\n";
			code += "   orbit_velocities *= texture(orbit_velocity_curve, vec2(lifetime)).rgb;\n";

			code += "	orbit_velocities *= degree_to_rad;\n";
			code += "	orbit_velocities *= delta/total_lifetime; // we wanna process those by the delta angle\n";
			code += "	//vec3 local_velocity_pivot = ((emission_transform) * vec4(velocity_pivot,1.0)).xyz;\n";
			code += "	// X axis\n";
			code += "	vec3 local_pos = (inverse(emission_transform) * transform[3]).xyz;\n";
			code += "	local_pos -= velocity_pivot;\n";
			code += "	local_pos.x = 0.;\n";
			code += "	mat3 x_rotation_mat = mat3(\n";
			code += "		vec3(1.0,0.0,0.0),\n";
			code += "		vec3(0.0, cos(orbit_velocities.x), sin(orbit_velocities.x)),\n";
			code += "		vec3(0.0, -sin(orbit_velocities.x), cos(orbit_velocities.x))\n";
			code += "	);\n";
			code += "	vec3 new_pos = x_rotation_mat * local_pos;\n";
			code += "	displacement = new_pos - local_pos;\n";
			code += "\n";
			code += "	// Y axis\n";
			code += "	local_pos = (inverse(emission_transform) * transform[3]).xyz;\n";
			code += "	local_pos -= velocity_pivot;\n";
			code += "	local_pos.y = 0.;\n";
			code += "	mat3 y_rotation_mat = mat3(\n";
			code += "		vec3(cos(orbit_velocities.y), 0.0, -sin(orbit_velocities.y)),\n";
			code += "		vec3(0.0, 1.0,0.0),\n";
			code += "		vec3(sin(orbit_velocities.y), 0.0, cos(orbit_velocities.y))\n";
			code += "	);\n";
			code += "	new_pos = y_rotation_mat * local_pos;\n";
			code += "	displacement += new_pos - local_pos;\n";
			code += "	// z axis\n";
			code += "\n";
			code += "	local_pos = (inverse(emission_transform) * transform[3]).xyz;\n";
			code += "	local_pos -= velocity_pivot;\n";
			code += "	local_pos.z = 0.;\n";
			code += "	mat3 z_rotation_mat = mat3(\n";
			code += "		vec3(cos(orbit_velocities.z),-sin(orbit_velocities.z),0.0),\n";
			code += "		vec3(-sin(orbit_velocities.z),cos(orbit_velocities.z), 0.0),\n";
			code += "		vec3(0.0,0.0,1.0)\n";
			code += "	);\n";
			code += "	new_pos = z_rotation_mat * local_pos;\n";
			code += "	displacement += new_pos - local_pos;\n";
			code += "\n";
		}
		code += "       return (emission_transform * vec4(displacement/delta, 0.0)).xyz;\n";
		code += "}\n";
		code += "\n";
		code += "\n";
	}

	code += "vec3 get_random_direction_from_spread(inout uint alt_seed, float spread_angle){\n";
	code += "	float pi = 3.14159;\n";
	code += "	float degree_to_rad = pi / 180.0;\n";
	code += "	vec3 velocity = vec3(0.);\n";
	code += "	float spread_rad = spread_angle * degree_to_rad;\n";
	code += "	float angle1_rad = rand_from_seed_m1_p1(alt_seed) * spread_rad;\n";
	code += "	float angle2_rad = rand_from_seed_m1_p1(alt_seed) * spread_rad * (1.0 - flatness);\n";
	code += "	vec3 direction_xz = vec3(sin(angle1_rad), 0.0, cos(angle1_rad));\n";
	code += "	vec3 direction_yz = vec3(0.0, sin(angle2_rad), cos(angle2_rad));\n";
	code += "	direction_yz.z = direction_yz.z / max(0.0001,sqrt(abs(direction_yz.z))); // better uniform distribution\n";
	code += "	vec3 spread_direction = vec3(direction_xz.x * direction_yz.z, direction_yz.y, direction_xz.z * direction_yz.z);\n";
	code += "	vec3 direction_nrm = length(direction) > 0.0 ? normalize(direction) : vec3(0.0, 0.0, 1.0);\n";
	code += "	// rotate spread to direction\n";
	code += "	vec3 binormal = cross(vec3(0.0, 1.0, 0.0), direction_nrm);\n";
	code += "	if (length(binormal) < 0.0001) {\n";
	code += "		// direction is parallel to Y. Choose Z as the binormal.\n";
	code += "		binormal = vec3(0.0, 0.0, 1.0);\n";
	code += "	}\n";
	code += "	binormal = normalize(binormal);\n";
	code += "	vec3 normal = cross(binormal, direction_nrm);\n";
	code += "	spread_direction = binormal * spread_direction.x + normal * spread_direction.y + direction_nrm * spread_direction.z;\n";
	code += "	return spread_direction;\n";

	code += "}\n";

	code += "vec3 process_radial_displacement(DynamicsParameters param, float lifetime, inout uint alt_seed, mat4 transform, mat4 emission_transform){\n";
	code += "	vec3 radial_displacement = vec3(0.0);\n";
	code += "	float radial_displacement_multiplier = 1.0;\n";
	if (tex_parameters[PARAM_RADIAL_VELOCITY].is_valid()) {
		code += "   radial_displacement_multiplier = texture(radial_velocity_curve, vec2(lifetime)).r;\n";
	}
	code += "	vec3 global_pivot = (emission_transform * vec4(velocity_pivot, 1.0)).xyz;\n";
	code += "	if(length(transform[3].xyz - global_pivot) > 0.01){\n";
	code += "		radial_displacement = normalize(transform[3].xyz - global_pivot) * radial_displacement_multiplier * param.radial_velocity;\n";
	code += "	}else{radial_displacement = get_random_direction_from_spread(alt_seed, 360.0)* param.radial_velocity;} \n";
	code += "	if (radial_displacement_multiplier * param.radial_velocity < 0.0){\n // Prevent inwards velocity to flicker once the point is reached.";
	code += "		if (length(radial_displacement) > 0.01){\n";
	code += "		radial_displacement = normalize(radial_displacement) * min(abs((radial_displacement_multiplier * param.radial_velocity)), length(transform[3].xyz - global_pivot));\n";
	code += "		}\n";
	code += "	\n";
	code += "	return radial_displacement;\n";
	code += "}\n";
	if (tex_parameters[PARAM_DIRECTIONAL_VELOCITY].is_valid()) {
		code += "vec3 process_directional_displacement(DynamicsParameters param, float lifetime_percent,mat4 transform, mat4 emission_transform){\n";
		code += "	vec3 displacement = vec3(0.);\n";
		if (directional_velocity_global) {
			code += "		displacement = texture(directional_velocity_curve, vec2(lifetime_percent)).xyz * param.directional_velocity;\n";
			code += "		displacement = (emission_transform * vec4(displacement, 0.0)).xyz;\n";
		} else {
			code += "		displacement = texture(directional_velocity_curve, vec2(lifetime_percent)).xyz * param.directional_velocity;\n";
		}
		code += "	return displacement;\n";
		code += "}\n";
	}

	code += "\n";
	code += "void process_physical_parameters(inout PhysicalParameters params, float lifetime_percent){\n";
	if (tex_parameters[PARAM_LINEAR_ACCEL].is_valid()) {
		code += "	params.linear_accel *= texture(linear_accel_texture, vec2(lifetime_percent)).r;\n";
	}
	if (tex_parameters[PARAM_RADIAL_ACCEL].is_valid()) {
		code += "	params.radial_accel *= texture(radial_accel_texture, vec2(lifetime_percent)).r;\n";
	}
	if (tex_parameters[PARAM_TANGENTIAL_ACCEL].is_valid()) {
		code += "	params.tangent_accel *= texture(tangent_accel_texture, vec2(lifetime_percent)).r;\n";
	}
	if (tex_parameters[PARAM_DAMPING].is_valid()) {
		code += "	params.damping *= texture(damping_texture, vec2(lifetime_percent)).r;\n";
	}
	code += "	\n";
	code += "}\n";
	code += "\n";

	code += "void start() {\n";
	code += "	uint base_number = NUMBER;\n";
	code += "	uint alt_seed = hash(base_number + uint(1) + RANDOM_SEED);\n";
	code += "	DisplayParameters params;\n";
	code += "	calculate_initial_display_params(params, alt_seed);\n";
	code += "	// reset alt seed?\n";
	code += "	// alt_seed = hash(base_number + uint(1) + RANDOM_SEED);\n";
	code += "	DynamicsParameters dynamic_params;\n";
	code += "	calculate_initial_dynamics_params(dynamic_params, alt_seed);\n";
	code += "	PhysicalParameters physics_params;\n";
	code += "	calculate_initial_physical_params(physics_params, alt_seed);\n";
	code += "   process_display_param(params, 0.0);\n";
	code += "	if (rand_from_seed(alt_seed) > AMOUNT_RATIO) {\n";
	code += "		ACTIVE = false;\n";
	code += "	}\n";
	code += "	\n";
	code += "	float pi = 3.14159;\n";
	code += "	float degree_to_rad = pi / 180.0;\n";
	code += "	\n";
	code += "	if (RESTART_CUSTOM){\n";
	code += "		CUSTOM = vec4(0.);\n";
	code += "		CUSTOM.w = params.lifetime;\n";
	code += "		CUSTOM.x = dynamic_params.angle;\n";
	code += "	}\n";
	code += "	if (RESTART_COLOR){\n";
	code += "		COLOR = params.color;\n";
	code += "	}\n";
	code += "	if (RESTART_ROT_SCALE) {\n";
	code += "		TRANSFORM[0].xyz = vec3(1.0, 0.0, 0.0);\n";
	code += "		TRANSFORM[1].xyz = vec3(0.0, 1.0, 0.0);\n";
	code += "		TRANSFORM[2].xyz = vec3(0.0, 0.0, 1.0);\n";
	code += "	}\n";
	code += "\n";
	code += "	if (RESTART_POSITION) {\n";
	code += "		TRANSFORM[3].xyz = calculate_initial_position(alt_seed);\n";
	if (turbulence_enabled) {
		code += "	float initial_turbulence_displacement = mix(turbulence_initial_displacement_min, turbulence_initial_displacement_max, rand_from_seed(alt_seed));\n";
		code += "			vec3 noise_direction = get_noise_direction(TRANSFORM[3].xyz);\n";
		code += "			TRANSFORM[3].xyz += noise_direction * initial_turbulence_displacement;\n";
	}
	code += "		TRANSFORM = EMISSION_TRANSFORM * TRANSFORM;\n";
	code += "		}\n";
	code += "	if (RESTART_VELOCITY) {\n";
	code += "		VELOCITY = get_random_direction_from_spread(alt_seed, spread) * dynamic_params.initial_velocity_multiplier;\n";
	code += "		}\n";
	code += "	process_display_param(params, 0.);\n";
	code += "//	process_dynamic_parameters(dynamic_params, 0., alt_seed, TRANSFORM, EMISSION_TRANSFORM, DELTA);\n";
	code += "	VELOCITY = (EMISSION_TRANSFORM * vec4(VELOCITY, 0.0)).xyz;\n";
	code += "	VELOCITY += EMITTER_VELOCITY * inherit_emitter_velocity_ratio;\n";
	if (particle_flags[PARTICLE_FLAG_DISABLE_Z]) {
		code += "		VELOCITY.z = 0.;\n";
		code += "		TRANSFORM[3].z = 0.;\n";
	}
	code += "}\n";
	code += "\n";

	code += "void process() {\n";
	code += "	uint base_number = NUMBER;\n";
	// TODO add optional determinism here
	code += "//	if (repeatable){\n";
	code += "//		base_number = INDEX;\n";
	code += "//	}\n";
	code += "	uint alt_seed = hash(base_number + uint(1) + RANDOM_SEED);\n";
	code += "	DisplayParameters params;\n";
	code += "	calculate_initial_display_params(params, alt_seed);\n";
	code += "	DynamicsParameters dynamic_params;\n";
	code += "	calculate_initial_dynamics_params(dynamic_params, alt_seed);\n";
	code += "	PhysicalParameters physics_params;\n";
	code += "	calculate_initial_physical_params(physics_params, alt_seed);\n";
	code += "	float pi = 3.14159;\n";
	code += "	float degree_to_rad = pi / 180.0;\n";
	code += "\n";
	code += "	CUSTOM.y += DELTA / LIFETIME;\n";
	code += "	CUSTOM.y = mix(CUSTOM.y, 1.0, INTERPOLATE_TO_END);\n";
	code += "	float lifetime_percent = CUSTOM.y/ params.lifetime;\n";
	code += "	if (CUSTOM.y > CUSTOM.w) {\n";
	code += "		ACTIVE = false;\n";
	code += "	}\n";
	code += "	\n";
	code += "	\n";
	code += "	\n";
	code += "	// will use this later to calculate final displacement and orient the particle.\n";
	code += "	vec3 starting_position = TRANSFORM[3].xyz;\n";
	code += "	vec3 controlled_displacement = vec3(0.0);\n";
	code += "	\n";
	code += "//	VELOCITY += process_physics_parameters(dynamic_params, lifetime_percent, alt_seed, TRANSFORM, EMISSION_TRANSFORM, DELTA);\n";
	code += "	\n";
	if (tex_parameters[PARAM_ORBIT_VELOCITY].is_valid() || particle_flags[PARTICLE_FLAG_DISABLE_Z]) {
		code += "	controlled_displacement += process_orbit_displacement(dynamic_params, lifetime_percent, alt_seed, TRANSFORM, EMISSION_TRANSFORM, DELTA, params.lifetime * LIFETIME);\n";
	}
	code += "	// calculate all velocity\n";
	code += "	\n";
	code += "	controlled_displacement += process_radial_displacement(dynamic_params, lifetime_percent, alt_seed, TRANSFORM, EMISSION_TRANSFORM);\n";
	code += "	\n";
	if (tex_parameters[PARAM_DIRECTIONAL_VELOCITY].is_valid()) {
		code += "	controlled_displacement += process_directional_displacement(dynamic_params, lifetime_percent, TRANSFORM, EMISSION_TRANSFORM);\n";
	}
	code += "	\n";
	code += "	process_physical_parameters(physics_params, lifetime_percent);\n";
	code += "	vec3 force;\n";
	code += "	{\n";
	code += "		// copied from previous version\n";
	code += "		vec3 pos = TRANSFORM[3].xyz;\n";
	code += "		force = gravity;\n";
	code += "		// apply linear acceleration\n";
	code += "		force += length(VELOCITY) > 0.0 ? normalize(VELOCITY) * physics_params.linear_accel : vec3(0.0);\n";
	code += "		// apply radial acceleration\n";
	code += "		vec3 org = EMISSION_TRANSFORM[3].xyz;\n";
	code += "		vec3 diff = pos - org;\n";
	code += "		force += length(diff) > 0.0 ? normalize(diff) * physics_params.radial_accel : vec3(0.0);\n";
	code += "		// apply tangential acceleration;\n";
	code += "		float tangent_accel_val = physics_params.tangent_accel;\n";
	if (particle_flags[PARTICLE_FLAG_DISABLE_Z]) {
		code += "       force += length(diff.yx) > 0.0 ? vec3(normalize(diff.yx * vec2(-1.0, 1.0)), 0.0) * tangent_accel_val : vec3(0.0);\n";
	} else {
		code += "		vec3 crossDiff = cross(normalize(diff), normalize(gravity));\n";
		code += "		force += length(crossDiff) > 0.0 ? normalize(crossDiff) * tangent_accel_val : vec3(0.0);\n";
	}
	if (attractor_interaction_enabled) {
		code += "		force += ATTRACTOR_FORCE;\n";
	}
	code += "\n";
	code += "		// apply attractor forces\n";
	if (particle_flags[PARTICLE_FLAG_DISABLE_Z]) {
		code += "			force.z = 0.;\n";
	}
	code += "		VELOCITY += force * DELTA;\n";
	code += "	}\n";
	code += "	{\n";
	code += "		// copied from previous version\n";
	code += "		if (physics_params.damping > 0.0) {\n";
	if (!particle_flags[PARTICLE_FLAG_DAMPING_AS_FRICTION]) {
		code += "				float v = length(VELOCITY);\n";
		code += "				v -= physics_params.damping * DELTA;\n";
		code += "				if (v < 0.0) {\n";
		code += "					VELOCITY = vec3(0.0);\n";
		code += "				} else {\n";
		code += "					VELOCITY = normalize(VELOCITY) * v;\n";
		code += "				}\n";
	} else {
		code += "				if (length(VELOCITY) > 0.01){\n";
		code += "					VELOCITY -= normalize(VELOCITY) * length(VELOCITY) * (physics_params.damping) * DELTA;\n";
		code += "				}\n";
	}
	code += "		}\n";
	code += "		\n";
	code += "	}\n";
	code += "	\n";
	if (collision_mode == COLLISION_RIGID) {
		code += "	if (COLLIDED) {\n";
		code += "		if (length(VELOCITY) > 3.0) {\n";
		code += "			TRANSFORM[3].xyz += COLLISION_NORMAL * COLLISION_DEPTH;\n";
		code += "			VELOCITY -= COLLISION_NORMAL * dot(COLLISION_NORMAL, VELOCITY) * (1.0 + collision_bounce);\n";
		code += "			VELOCITY = mix(VELOCITY,vec3(0.0),clamp(collision_friction, 0.0, 1.0));\n";
		code += "		} else {\n";
		code += "			VELOCITY = vec3(0.0);\n";
		// If turbulence is enabled, set the noise direction to up so the turbulence color is "neutral"
		if (turbulence_enabled) {
			code += "			noise_direction = vec3(1.0, 0.0, 0.0);\n";
		}
		code += "		}\n";
		code += "	}\n";
	} else if (collision_mode == COLLISION_HIDE_ON_CONTACT) {
		code += "	if (COLLIDED) {\n";
		code += "		ACTIVE = false;\n";
		code += "	}\n";
	}
	code += "	vec3 final_velocity = controlled_displacement + VELOCITY;\n";
	code += "	\n";
	code += "	// turbulence before limiting\n";
	if (turbulence_enabled) {
		if (tex_parameters[PARAM_TURB_INFLUENCE_OVER_LIFE].is_valid()) {
			code += "		float turbulence_influence = textureLod(turbulence_influence_over_life, vec2(lifetime_percent, 0.0), 0.0).r;\n";
		} else {
			code += "   float turbulence_influence = 1.0;\n";
		}
		code += "		\n";
		code += "		vec3 noise_direction = get_noise_direction(TRANSFORM[3].xyz);\n";
		code += "		if (!COLLIDED) {\n";
		code += "			\n";
		code += "			float vel_mag = length(final_velocity);\n";
		code += "			float vel_infl = clamp(dynamic_params.turb_influence * turbulence_influence, 0.0,1.0);\n";
		code += "			final_velocity = mix(final_velocity, normalize(noise_direction) * vel_mag * (1.0 + (1.0 - vel_infl) * 0.2), vel_infl);\n";
		code += "		}\n";
	}
	code += "	\n";
	code += "	// limit velocity\n";
	if (velocity_limit_curve.is_valid()) {
		code += "	if (length(final_velocity) > 0.001){\n";
		code += "		final_velocity = normalize(final_velocity) * min(abs(length(final_velocity)), abs(texture(velocity_limit_curve, vec2(lifetime_percent)).r));\n";
		code += "	}\n";
	}
	code += "	\n";
	if (particle_flags[PARTICLE_FLAG_DISABLE_Z]) {
		code += "		final_velocity.z = 0.;\n";
	}
	code += "	TRANSFORM[3].xyz += final_velocity * DELTA;\n";
	code += "	\n";
	code += "	\n";
	code += "	process_display_param(params, lifetime_percent);\n";
	code += "	\n";
	code += "	float base_angle = dynamic_params.angle;\n";
	if (tex_parameters[PARAM_ANGLE].is_valid()) {
		code += "	base_angle *= texture(angle_texture, vec2(lifetime_percent)).r;\n";
	}
	if (tex_parameters[PARAM_ANGULAR_VELOCITY].is_valid()) {
		code += "	base_angle += CUSTOM.y * LIFETIME * dynamic_params.angular_velocity * texture(angular_velocity_texture, vec2(lifetime_percent)).r;\n";

	} else {
		code += "	base_angle += CUSTOM.y * LIFETIME * dynamic_params.angular_velocity;\n";
	}
	code += "	CUSTOM.x = base_angle * degree_to_rad;\n";
	code += "   COLOR = params.color;\n";

	if (particle_flags[PARTICLE_FLAG_DISABLE_Z]) {
		if (particle_flags[PARTICLE_FLAG_ALIGN_Y_TO_VELOCITY]) {
			code += "	if (length(final_velocity) > 0.0) {\n";
			code += "		TRANSFORM[1].xyz = normalize(final_velocity);\n";
			code += "	} else {\n";
			code += "		TRANSFORM[1].xyz = normalize(TRANSFORM[1].xyz);\n";
			code += "	}\n";
			code += "	TRANSFORM[0].xyz = normalize(cross(TRANSFORM[1].xyz, TRANSFORM[2].xyz));\n";
			code += "	TRANSFORM[2] = vec4(0.0, 0.0, 1.0, 0.0);\n";
		} else {
			code += "	TRANSFORM[0] = vec4(cos(CUSTOM.x), -sin(CUSTOM.x), 0.0, 0.0);\n";
			code += "	TRANSFORM[1] = vec4(sin(CUSTOM.x), cos(CUSTOM.x), 0.0, 0.0);\n";
			code += "	TRANSFORM[2] = vec4(0.0, 0.0, 1.0, 0.0);\n";
		}

	} else {
		// orient particle Y towards velocity
		if (particle_flags[PARTICLE_FLAG_ALIGN_Y_TO_VELOCITY]) {
			code += "	if (length(final_velocity) > 0.0) {\n";
			code += "		TRANSFORM[1].xyz = normalize(final_velocity);\n";
			code += "	} else {\n";
			code += "		TRANSFORM[1].xyz = normalize(TRANSFORM[1].xyz);\n";
			code += "	}\n";
			code += "	if (TRANSFORM[1].xyz == normalize(TRANSFORM[0].xyz)) {\n";
			code += "		TRANSFORM[0].xyz = normalize(cross(normalize(TRANSFORM[1].xyz), normalize(TRANSFORM[2].xyz)));\n";
			code += "		TRANSFORM[2].xyz = normalize(cross(normalize(TRANSFORM[0].xyz), normalize(TRANSFORM[1].xyz)));\n";
			code += "	} else {\n";
			code += "		TRANSFORM[2].xyz = normalize(cross(normalize(TRANSFORM[0].xyz), normalize(TRANSFORM[1].xyz)));\n";
			code += "		TRANSFORM[0].xyz = normalize(cross(normalize(TRANSFORM[1].xyz), normalize(TRANSFORM[2].xyz)));\n";
			code += "	}\n";
		} else {
			code += "	TRANSFORM[0].xyz = normalize(TRANSFORM[0].xyz);\n";
			code += "	TRANSFORM[1].xyz = normalize(TRANSFORM[1].xyz);\n";
			code += "	TRANSFORM[2].xyz = normalize(TRANSFORM[2].xyz);\n";
		}
		// turn particle by rotation in Y
		if (particle_flags[PARTICLE_FLAG_ROTATE_Y]) {
			code += "	vec4 origin = TRANSFORM[3];\n";
			code += "	TRANSFORM = mat4(vec4(cos(CUSTOM.x), 0.0, -sin(CUSTOM.x), 0.0), vec4(0.0, 1.0, 0.0, 0.0), vec4(sin(CUSTOM.x), 0.0, cos(CUSTOM.x), 0.0), vec4(0.0, 0.0, 0.0, 1.0));\n";
			code += "	TRANSFORM[3] = origin;\n";
		}
	}

	if (particle_flags[PARTICLE_FLAG_DISABLE_Z]) {
		code += "	TRANSFORM[3].z = 0.0;\n";
	}
	if (tex_parameters[PARAM_SCALE_OVER_VELOCITY].is_valid()) {
		code += "	if(length(final_velocity) > 0.001){\n";
		code += "		params.scale *= texture(scale_over_velocity_curve, vec2(clamp(length(final_velocity)/(scale_over_velocity_max - scale_over_velocity_min), 0.0,1.0), 0.0)).rgb;\n";
		code += "	} else {params.scale *= texture(scale_over_velocity_curve, vec2(0.0)).rgb;}\n \n";
	}
	code += "//	params.scale *= length(final_velocity)/100.0;\n";
	code += "\n";
	code += "	TRANSFORM[0].xyz *= sign(params.scale.x) * max(abs(params.scale.x), 0.001);\n";
	code += "	TRANSFORM[1].xyz *= sign(params.scale.y) * max(abs(params.scale.y), 0.001);\n";
	code += "	TRANSFORM[2].xyz *= sign(params.scale.z) * max(abs(params.scale.z), 0.001);\n";
	code += "	\n";
	code += "	// \n";
	code += "	CUSTOM.z = params.animation_offset + lifetime_percent * params.animation_speed;\n";
	code += "	\n";

	if (sub_emitter_mode != SUB_EMITTER_DISABLED && !RenderingServer::get_singleton()->is_low_end()) {
		code += "	int emit_count = 0;\n";
		switch (sub_emitter_mode) {
			case SUB_EMITTER_CONSTANT: {
				code += "	float interval_from = CUSTOM.y * LIFETIME - DELTA;\n";
				code += "	float interval_rem = sub_emitter_frequency - mod(interval_from,sub_emitter_frequency);\n";
				code += "	if (DELTA >= interval_rem) emit_count = 1;\n";
			} break;
			case SUB_EMITTER_AT_COLLISION: {
				code += "	if (COLLIDED) emit_count = sub_emitter_amount_at_collision;\n";
			} break;
			case SUB_EMITTER_AT_END: {
				code += "	float unit_delta = DELTA/LIFETIME;\n";
				code += "	float end_time = CUSTOM.w * 0.95;\n"; // if we do at the end we might miss it, as it can just get deactivated by emitter
				code += "	if (CUSTOM.y < end_time && (CUSTOM.y + unit_delta) >= end_time) emit_count = sub_emitter_amount_at_end;\n";
			} break;
			default: {
			}
		}
		code += "	for(int i=0;i<emit_count;i++) {\n";
		code += "		uint flags = FLAG_EMIT_POSITION|FLAG_EMIT_ROT_SCALE;\n";
		code += "		if (sub_emitter_keep_velocity) flags|=FLAG_EMIT_VELOCITY;\n";
		code += "		emit_subparticle(TRANSFORM,VELOCITY,vec4(0.0),vec4(0.0),flags);\n";
		code += "	}";
	}

	code += "	if (CUSTOM.y > CUSTOM.w) {\n";
	code += "		ACTIVE = false;\n";
	code += "	}\n";
	code += "}\n";
	code += "\n";

	ShaderData shader_data;
	shader_data.shader = RS::get_singleton()->shader_create();
	shader_data.users = 1;

	RS::get_singleton()->shader_set_code(shader_data.shader, code);

	shader_map[mk] = shader_data;

	RS::get_singleton()->material_set_shader(_get_material(), shader_data.shader);
}
