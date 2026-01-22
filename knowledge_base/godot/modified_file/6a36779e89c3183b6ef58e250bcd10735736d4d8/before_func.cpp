		default: {
			return !is_zero();
		}
	}

	return false;
}

void Variant::reference(const Variant &p_variant) {

	clear();

	type = p_variant.type;

	switch (p_variant.type) {
		case NIL: {

			// none
		} break;

		// atomic types
		case BOOL: {

			_data._bool = p_variant._data._bool;
		} break;
		case INT: {

			_data._int = p_variant._data._int;
		} break;
		case REAL: {

			_data._real = p_variant._data._real;
		} break;
		case STRING: {

			memnew_placement(_data._mem, String(*reinterpret_cast<const String *>(p_variant._data._mem)));
		} break;

		// math types
		case VECTOR2: {

			memnew_placement(_data._mem, Vector2(*reinterpret_cast<const Vector2 *>(p_variant._data._mem)));
		} break;
		case RECT2: {

			memnew_placement(_data._mem, Rect2(*reinterpret_cast<const Rect2 *>(p_variant._data._mem)));
		} break;
		case TRANSFORM2D: {

			_data._transform2d = memnew(Transform2D(*p_variant._data._transform2d));
		} break;
		case VECTOR3: {

			memnew_placement(_data._mem, Vector3(*reinterpret_cast<const Vector3 *>(p_variant._data._mem)));
		} break;
		case PLANE: {

			memnew_placement(_data._mem, Plane(*reinterpret_cast<const Plane *>(p_variant._data._mem)));
		} break;

		case AABB: {

			_data._aabb = memnew(::AABB(*p_variant._data._aabb));
		} break;
		case QUAT: {

			memnew_placement(_data._mem, Quat(*reinterpret_cast<const Quat *>(p_variant._data._mem)));

		} break;
		case BASIS: {

			_data._basis = memnew(Basis(*p_variant._data._basis));

		} break;
		case TRANSFORM: {

			_data._transform = memnew(Transform(*p_variant._data._transform));
		} break;

		// misc types
		case COLOR: {

			memnew_placement(_data._mem, Color(*reinterpret_cast<const Color *>(p_variant._data._mem)));

		} break;
		case _RID: {

			memnew_placement(_data._mem, RID(*reinterpret_cast<const RID *>(p_variant._data._mem)));
		} break;
		case OBJECT: {

			memnew_placement(_data._mem, ObjData(p_variant._get_obj()));
		} break;
		case NODE_PATH: {

			memnew_placement(_data._mem, NodePath(*reinterpret_cast<const NodePath *>(p_variant._data._mem)));

		} break;
		case DICTIONARY: {

			memnew_placement(_data._mem, Dictionary(*reinterpret_cast<const Dictionary *>(p_variant._data._mem)));

		} break;
		case ARRAY: {

			memnew_placement(_data._mem, Array(*reinterpret_cast<const Array *>(p_variant._data._mem)));

		} break;

		// arrays
		case POOL_BYTE_ARRAY: {

			memnew_placement(_data._mem, PoolVector<uint8_t>(*reinterpret_cast<const PoolVector<uint8_t> *>(p_variant._data._mem)));

		} break;
		case POOL_INT_ARRAY: {

			memnew_placement(_data._mem, PoolVector<int>(*reinterpret_cast<const PoolVector<int> *>(p_variant._data._mem)));

		} break;
		case POOL_REAL_ARRAY: {

			memnew_placement(_data._mem, PoolVector<real_t>(*reinterpret_cast<const PoolVector<real_t> *>(p_variant._data._mem)));

		} break;
		case POOL_STRING_ARRAY: {

			memnew_placement(_data._mem, PoolVector<String>(*reinterpret_cast<const PoolVector<String> *>(p_variant._data._mem)));

		} break;
		case POOL_VECTOR2_ARRAY: {

			memnew_placement(_data._mem, PoolVector<Vector2>(*reinterpret_cast<const PoolVector<Vector2> *>(p_variant._data._mem)));

		} break;
		case POOL_VECTOR3_ARRAY: {

			memnew_placement(_data._mem, PoolVector<Vector3>(*reinterpret_cast<const PoolVector<Vector3> *>(p_variant._data._mem)));

		} break;
		case POOL_COLOR_ARRAY: {
