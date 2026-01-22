validity_t *ColumnDataCollectionSegment::GetValidityPointer(data_ptr_t base_ptr, idx_t type_size, idx_t count) {
	auto validity_mask = reinterpret_cast<validity_t *>(base_ptr + GetDataSize(type_size));

	// Optimized check to see if all entries are valid
	for (idx_t i = 0; i < (count / ValidityMask::BITS_PER_VALUE); i++) {
		if (!ValidityMask::AllValid(validity_mask[i])) {
			return validity_mask;
		}
	}

	if ((count % ValidityMask::BITS_PER_VALUE) != 0) {
		// Create a mask with the lower `bits_to_check` bits set to 1
		validity_t mask = (1ULL << (count % ValidityMask::BITS_PER_VALUE)) - 1;
		if ((validity_mask[(count / ValidityMask::BITS_PER_VALUE)] & mask) != mask) {
			return validity_mask;
		}
	}
	// All entries are valid, no need to initialize the validity mask
	return nullptr;
}
