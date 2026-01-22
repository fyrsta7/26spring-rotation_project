idx_t GroupedAggregateHashTable::FindOrCreateGroupsInternal(DataChunk &groups, Vector &group_hashes_v,
                                                            Vector &addresses_v, SelectionVector &new_groups_out) {
	D_ASSERT(groups.ColumnCount() + 1 == layout.ColumnCount());
	D_ASSERT(group_hashes_v.GetType() == LogicalType::HASH);
	D_ASSERT(state.ht_offsets.GetVectorType() == VectorType::FLAT_VECTOR);
	D_ASSERT(state.ht_offsets.GetType() == LogicalType::UBIGINT);
	D_ASSERT(addresses_v.GetType() == LogicalType::POINTER);
	D_ASSERT(state.hash_salts.GetType() == LogicalType::HASH);

	// Need to fit the entire vector, and resize at threshold
	const auto chunk_size = groups.size();
	if (Count() + chunk_size > capacity || Count() + chunk_size > ResizeThreshold()) {
		Verify();
		Resize(capacity * 2);
	}
	D_ASSERT(capacity - Count() >= chunk_size); // we need to be able to fit at least one vector of data

	// we start out with all entries [0, 1, 2, ..., chunk_size]
	const SelectionVector *sel_vector = FlatVector::IncrementalSelectionVector();

	// Make a chunk that references the groups and the hashes and convert to unified format
	if (state.group_chunk.ColumnCount() == 0) {
		state.group_chunk.InitializeEmpty(layout.GetTypes());
	}
	D_ASSERT(state.group_chunk.ColumnCount() == layout.GetTypes().size());
	for (idx_t grp_idx = 0; grp_idx < groups.ColumnCount(); grp_idx++) {
		state.group_chunk.data[grp_idx].Reference(groups.data[grp_idx]);
	}
	state.group_chunk.data[groups.ColumnCount()].Reference(group_hashes_v);
	state.group_chunk.SetCardinality(groups);

	// convert all vectors to unified format
	TupleDataCollection::ToUnifiedFormat(state.partitioned_append_state.chunk_state, state.group_chunk);
	if (!state.group_data) {
		state.group_data = make_unsafe_uniq_array_uninitialized<UnifiedVectorFormat>(state.group_chunk.ColumnCount());
	}
	TupleDataCollection::GetVectorData(state.partitioned_append_state.chunk_state, state.group_data.get());

	group_hashes_v.Flatten(chunk_size);
	const auto hashes = FlatVector::GetData<hash_t>(group_hashes_v);

	addresses_v.Flatten(chunk_size);
	const auto addresses = FlatVector::GetData<data_ptr_t>(addresses_v);

	if (skip_lookups) {
		// Just appending now
		partitioned_data->AppendUnified(state.partitioned_append_state, state.group_chunk,
		                                *FlatVector::IncrementalSelectionVector(), chunk_size);
		RowOperations::InitializeStates(layout, state.partitioned_append_state.chunk_state.row_locations,
		                                *FlatVector::IncrementalSelectionVector(), chunk_size);

		const auto row_locations =
		    FlatVector::GetData<data_ptr_t>(state.partitioned_append_state.chunk_state.row_locations);
		const auto &row_sel = state.partitioned_append_state.reverse_partition_sel;
		for (idx_t i = 0; i < chunk_size; i++) {
			const auto &row_idx = row_sel[i];
			const auto &row_location = row_locations[row_idx];
			addresses[i] = row_location;
		}
		count += chunk_size;
		return chunk_size;
	}

	// Compute the entry in the table based on the hash using a modulo,
	// and precompute the hash salts for faster comparison below
	const auto ht_offsets = FlatVector::GetData<uint64_t>(state.ht_offsets);
	const auto hash_salts = FlatVector::GetData<hash_t>(state.hash_salts);

	// We also compute the occupied count, which is essentially useless.
	// However, this loop is branchless, while the main lookup loop below is not.
	// So, by doing the lookups here, we better amortize cache misses.
	idx_t occupied_count = 0;
	for (idx_t r = 0; r < chunk_size; r++) {
		const auto &hash = hashes[r];
		auto &ht_offset = ht_offsets[r];
		ht_offset = ApplyBitMask(hash);
		occupied_count += entries[ht_offset].IsOccupied(); // Lookup
		D_ASSERT(ht_offset == hash % capacity);
		hash_salts[r] = ht_entry_t::ExtractSalt(hash);
	}

	idx_t new_group_count = 0;
	idx_t remaining_entries = chunk_size;
	idx_t iteration_count;
	for (iteration_count = 0; remaining_entries > 0 && iteration_count < capacity; iteration_count++) {
		idx_t new_entry_count = 0;
		idx_t need_compare_count = 0;
		idx_t no_match_count = 0;

		// For each remaining entry, figure out whether or not it belongs to a full or empty group
		for (idx_t i = 0; i < remaining_entries; i++) {
			const auto index = sel_vector->get_index(i);
			const auto salt = hash_salts[index];
			auto &ht_offset = ht_offsets[index];

			idx_t inner_iteration_count;
			for (inner_iteration_count = 0; inner_iteration_count < capacity; inner_iteration_count++) {
				auto &entry = entries[ht_offset];
				if (!entry.IsOccupied()) { // Unoccupied: claim it
					entry.SetSalt(salt);
					state.empty_vector.set_index(new_entry_count++, index);
					new_groups_out.set_index(new_group_count++, index);
					break;
				}

				if (DUCKDB_LIKELY(entry.GetSalt() == salt)) { // Matching salt: compare groups
					state.group_compare_vector.set_index(need_compare_count++, index);
					break;
				}

				// Linear probing
				IncrementAndWrap(ht_offset, bitmask);
			}
			if (DUCKDB_UNLIKELY(inner_iteration_count == capacity)) {
				throw InternalException("Maximum inner iteration count reached in GroupedAggregateHashTable");
			}
		}

		if (DUCKDB_UNLIKELY(occupied_count > new_entry_count + need_compare_count)) {
			// We use the useless occupied_count we summed above here so the variable is used,
			// and the compiler cannot optimize away the vectorized lookups above. This should never be triggered.
			throw InternalException("Internal validation failed in GroupedAggregateHashTable");
		}
		occupied_count = 0; // Have to set to 0 for next iterations

		if (new_entry_count != 0) {
			// Append everything that belongs to an empty group
			optional_ptr<PartitionedTupleData> data;
			optional_ptr<PartitionedTupleDataAppendState> append_state;
			if (radix_bits >= UNPARTITIONED_RADIX_BITS_THRESHOLD &&
			    new_entry_count / RadixPartitioning::NumberOfPartitions(radix_bits) <= 4) {
				TupleDataCollection::ToUnifiedFormat(state.unpartitioned_append_state.chunk_state, state.group_chunk);
				data = unpartitioned_data.get();
				append_state = &state.unpartitioned_append_state;
			} else {
				data = partitioned_data.get();
				append_state = &state.partitioned_append_state;
			}
			data->AppendUnified(*append_state, state.group_chunk, state.empty_vector, new_entry_count);
			RowOperations::InitializeStates(layout, append_state->chunk_state.row_locations,
			                                *FlatVector::IncrementalSelectionVector(), new_entry_count);

			// Set the entry pointers in the 1st part of the HT now that the data has been appended
			const auto row_locations = FlatVector::GetData<data_ptr_t>(append_state->chunk_state.row_locations);
			const auto &row_sel = append_state->reverse_partition_sel;
			for (idx_t new_entry_idx = 0; new_entry_idx < new_entry_count; new_entry_idx++) {
				const auto &index = state.empty_vector[new_entry_idx];
				const auto &row_idx = row_sel[index];
				const auto &row_location = row_locations[row_idx];

				auto &entry = entries[ht_offsets[index]];

				entry.SetPointer(row_location);
				addresses[index] = row_location;
			}
		}

		if (need_compare_count != 0) {
			// Get the pointers to the rows that need to be compared
			for (idx_t need_compare_idx = 0; need_compare_idx < need_compare_count; need_compare_idx++) {
				const auto &index = state.group_compare_vector[need_compare_idx];
				const auto &entry = entries[ht_offsets[index]];
				addresses[index] = entry.GetPointer();
			}

			// Perform group comparisons
			row_matcher.Match(state.group_chunk, state.partitioned_append_state.chunk_state.vector_data,
			                  state.group_compare_vector, need_compare_count, layout, addresses_v,
			                  &state.no_match_vector, no_match_count);
		}

		// Linear probing: each of the entries that do not match move to the next entry in the HT
		for (idx_t i = 0; i < no_match_count; i++) {
			const auto &index = state.no_match_vector[i];
			auto &ht_offset = ht_offsets[index];
			IncrementAndWrap(ht_offset, bitmask);
		}
		sel_vector = &state.no_match_vector;
		remaining_entries = no_match_count;
	}
	if (iteration_count == capacity) {
		throw InternalException("Maximum outer iteration count reached in GroupedAggregateHashTable");
	}

	count += new_group_count;
	return new_group_count;
}
