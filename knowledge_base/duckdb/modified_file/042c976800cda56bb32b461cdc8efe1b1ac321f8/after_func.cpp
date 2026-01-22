template<class T>
static void MergeUpdateInfo(UpdateInfo *current, T *result_data, ValidityMask &result_mask) {
	ValidityMask current_mask(current->validity);
	auto info_data = (T *)current->tuple_data;

	auto info_data = (T *)current->tuple_data;
	if (current->N == STANDARD_VECTOR_SIZE) {
		// special case: update touches ALL tuples of this vector
		// in this case we can just memcpy the data
		// since the layout of the update info is guaranteed to be [0, 1, 2, 3, ...]
		memcpy(result_data, info_data, sizeof(T) * current->N);
		result_mask.EnsureWritable();
		memcpy(result_mask.GetData(), current_mask.GetData(), ValidityMask::STANDARD_MASK_SIZE);
	} else {
		for (idx_t i = 0; i < current->N; i++) {
			result_data[current->tuples[i]] = info_data[i];
			result_mask.Set(current->tuples[i], current_mask.RowIsValidUnsafe(i));
		}
	}
}
