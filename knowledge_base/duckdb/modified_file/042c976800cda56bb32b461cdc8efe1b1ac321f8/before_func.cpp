template<class T>
static void MergeUpdateInfo(UpdateInfo *current, T *result_data, ValidityMask &result_mask) {
	ValidityMask current_mask(current->validity);
	auto info_data = (T *)current->tuple_data;
	for (idx_t i = 0; i < current->N; i++) {
		result_data[current->tuples[i]] = info_data[i];
		result_mask.Set(current->tuples[i], current_mask.RowIsValidUnsafe(i));
	}
}
