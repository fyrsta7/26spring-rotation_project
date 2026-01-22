void FlatVector::SetNull(Vector &vector, idx_t idx, bool is_null) {
	D_ASSERT(vector.GetVectorType() == VectorType::FLAT_VECTOR);
	vector.validity.Set(idx, !is_null);
	if (is_null) {
		auto type = vector.GetType();
		auto internal_type = type.InternalType();
		if (internal_type == PhysicalType::STRUCT) {
			// set all child entries to null as well
			auto &entries = StructVector::GetEntries(vector);
			for (auto &entry : entries) {
				FlatVector::SetNull(*entry, idx, is_null);
			}
		} else if (internal_type == PhysicalType::ARRAY) {
			// set the child element in the array to null as well
			auto &child = ArrayVector::GetEntry(vector);
			auto array_size = ArrayType::GetSize(type);
			auto child_offset = idx * array_size;
			for (idx_t i = 0; i < array_size; i++) {
				FlatVector::SetNull(child, child_offset + i, is_null);
			}
		}
	}
}
