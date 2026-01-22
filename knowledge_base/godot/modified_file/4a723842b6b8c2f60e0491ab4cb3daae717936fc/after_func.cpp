                                                                        \
	for (int i = 0; i < l.size(); ++i) {                                \
		if (!p_compare_func((lr[i]), (rr[i])))                          \
			return false;                                               \
	}                                                                   \
                                                                        \
	return true

bool Variant::hash_compare(const Variant &p_variant) const {
	if (type != p_variant.type) {
		return false;
	}

	switch (type) {
		case INT: {
			return _data._int == p_variant._data._int;
		} break;

		case FLOAT: {
			return hash_compare_scalar(_data._float, p_variant._data._float);
		} break;

		case STRING: {
			return *reinterpret_cast<const String *>(_data._mem) == *reinterpret_cast<const String *>(p_variant._data._mem);
		} break;

		case VECTOR2: {
			const Vector2 *l = reinterpret_cast<const Vector2 *>(_data._mem);
			const Vector2 *r = reinterpret_cast<const Vector2 *>(p_variant._data._mem);

			return hash_compare_vector2(*l, *r);
		} break;
		case VECTOR2I: {
			const Vector2i *l = reinterpret_cast<const Vector2i *>(_data._mem);
			const Vector2i *r = reinterpret_cast<const Vector2i *>(p_variant._data._mem);
			return *l == *r;
		} break;

		case RECT2: {
			const Rect2 *l = reinterpret_cast<const Rect2 *>(_data._mem);
			const Rect2 *r = reinterpret_cast<const Rect2 *>(p_variant._data._mem);

			return (hash_compare_vector2(l->position, r->position)) &&
				   (hash_compare_vector2(l->size, r->size));
		} break;
		case RECT2I: {
			const Rect2i *l = reinterpret_cast<const Rect2i *>(_data._mem);
			const Rect2i *r = reinterpret_cast<const Rect2i *>(p_variant._data._mem);

			return *l == *r;
		} break;

		case TRANSFORM2D: {
			Transform2D *l = _data._transform2d;
			Transform2D *r = p_variant._data._transform2d;

			for (int i = 0; i < 3; i++) {
				if (!(hash_compare_vector2(l->elements[i], r->elements[i]))) {
					return false;
				}
			}

			return true;
		} break;

		case VECTOR3: {
			const Vector3 *l = reinterpret_cast<const Vector3 *>(_data._mem);
			const Vector3 *r = reinterpret_cast<const Vector3 *>(p_variant._data._mem);

			return hash_compare_vector3(*l, *r);
		} break;
		case VECTOR3I: {
			const Vector3i *l = reinterpret_cast<const Vector3i *>(_data._mem);
			const Vector3i *r = reinterpret_cast<const Vector3i *>(p_variant._data._mem);

			return *l == *r;
		} break;

		case PLANE: {
			const Plane *l = reinterpret_cast<const Plane *>(_data._mem);
			const Plane *r = reinterpret_cast<const Plane *>(p_variant._data._mem);

			return (hash_compare_vector3(l->normal, r->normal)) &&
				   (hash_compare_scalar(l->d, r->d));
		} break;

		case AABB: {
			const ::AABB *l = _data._aabb;
			const ::AABB *r = p_variant._data._aabb;

			return (hash_compare_vector3(l->position, r->position) &&
					(hash_compare_vector3(l->size, r->size)));

		} break;

		case QUATERNION: {
			const Quaternion *l = reinterpret_cast<const Quaternion *>(_data._mem);
			const Quaternion *r = reinterpret_cast<const Quaternion *>(p_variant._data._mem);

			return hash_compare_quaternion(*l, *r);
		} break;

		case BASIS: {
			const Basis *l = _data._basis;
			const Basis *r = p_variant._data._basis;

			for (int i = 0; i < 3; i++) {
				if (!(hash_compare_vector3(l->elements[i], r->elements[i]))) {
					return false;
				}
			}

			return true;
		} break;

		case TRANSFORM3D: {
			const Transform3D *l = _data._transform3d;
			const Transform3D *r = p_variant._data._transform3d;

			for (int i = 0; i < 3; i++) {
				if (!(hash_compare_vector3(l->basis.elements[i], r->basis.elements[i]))) {
					return false;
				}
			}

			return hash_compare_vector3(l->origin, r->origin);
		} break;

		case COLOR: {
			const Color *l = reinterpret_cast<const Color *>(_data._mem);
			const Color *r = reinterpret_cast<const Color *>(p_variant._data._mem);

			return hash_compare_color(*l, *r);
		} break;

		case ARRAY: {
			const Array &l = *(reinterpret_cast<const Array *>(_data._mem));
			const Array &r = *(reinterpret_cast<const Array *>(p_variant._data._mem));

			if (l.size() != r.size()) {
				return false;
			}

			for (int i = 0; i < l.size(); ++i) {
				if (!l[i].hash_compare(r[i])) {
					return false;
				}
			}

			return true;
		} break;

		// This is for floating point comparisons only.
		case PACKED_FLOAT32_ARRAY: {
			hash_compare_packed_array(_data.packed_array, p_variant._data.packed_array, float, hash_compare_scalar);
		} break;

		case PACKED_FLOAT64_ARRAY: {
			hash_compare_packed_array(_data.packed_array, p_variant._data.packed_array, double, hash_compare_scalar);
		} break;

		case PACKED_VECTOR2_ARRAY: {
			hash_compare_packed_array(_data.packed_array, p_variant._data.packed_array, Vector2, hash_compare_vector2);
		} break;

		case PACKED_VECTOR3_ARRAY: {
			hash_compare_packed_array(_data.packed_array, p_variant._data.packed_array, Vector3, hash_compare_vector3);
		} break;

		case PACKED_COLOR_ARRAY: {
			hash_compare_packed_array(_data.packed_array, p_variant._data.packed_array, Color, hash_compare_color);
		} break;

		default:
