std::pair<bool, list_entry_t> ListVector::GetConsecutiveChildList(Vector &list, Vector &result, idx_t offset,
                                                                  idx_t count) {

	UnifiedVectorFormat unified_list_data;
	list.ToUnifiedFormat(offset + count, unified_list_data);
	auto list_data = (list_entry_t *)unified_list_data.data;

	// boolean, if constant, and offset and length of the relevant child vector
	std::pair<bool, list_entry_t> info(true, list_entry_t(0, 0));

	// find the first non-NULL entry
	idx_t first_length = 0;
	for (idx_t i = offset; i < offset + count; i++) {
		auto idx = unified_list_data.sel->get_index(i);
		if (!unified_list_data.validity.RowIsValid(idx)) {
			continue;
		}
		info.second.offset = list_data[idx].offset;
		first_length = list_data[idx].length;
		break;
	}

	// small performance improvement for constant vectors
	// avoids iterating over all their (constant) elements
	if (list.GetVectorType() == VectorType::CONSTANT_VECTOR) {
		info.second.length = first_length;
		return info;
	}

	// now get the child count and determine whether the children are stored consecutively
	// also determine if a flat vector has pseudo constant values (all offsets + length the same)
	// this can happen e.g. for UNNESTs
	bool is_consecutive = true;
	for (idx_t i = offset; i < offset + count; i++) {
		auto idx = unified_list_data.sel->get_index(i);
		if (!unified_list_data.validity.RowIsValid(idx)) {
			continue;
		}
		if (list_data[idx].offset != info.second.offset || list_data[idx].length != first_length) {
			info.first = false;
		}
		if (list_data[idx].offset != info.second.offset + info.second.length) {
			is_consecutive = false;
		}
		info.second.length += list_data[idx].length;
	}

	if (info.first) {
		info.second.length = first_length;
	}

	if (!info.first && !is_consecutive) {
		SelectionVector child_sel(info.second.length);
		idx_t entry = 0;

		for (idx_t i = offset; i < offset + count; i++) {
			auto idx = unified_list_data.sel->get_index(i);
			if (!unified_list_data.validity.RowIsValid(idx)) {
				continue;
			}
			for (idx_t k = 0; k < list_data[idx].length; k++) {
				child_sel.set_index(entry++, list_data[idx].offset + k);
			}
		}

		result.Slice(child_sel, info.second.length);
		result.Flatten(info.second.length);
		info.second.offset = 0;
	}

	return info;
}
