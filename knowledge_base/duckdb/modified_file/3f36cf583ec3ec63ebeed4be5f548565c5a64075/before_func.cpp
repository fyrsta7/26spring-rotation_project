static void SortCollectionForPartition(WindowOperatorState &state, BoundWindowExpression *wexpr, ChunkCollection &input,
                                       ChunkCollection &over, ChunkCollection *hashes, const hash_t hash_bin,
                                       const hash_t hash_mask) {
	if (input.Count() == 0) {
		return;
	}

	vector<BoundOrderByNode> orders;
	// we sort by both 1) partition by expression list and 2) order by expressions
	for (idx_t prt_idx = 0; prt_idx < wexpr->partitions.size(); prt_idx++) {
		if (wexpr->partitions_stats.empty() || !wexpr->partitions_stats[prt_idx]) {
			orders.emplace_back(OrderType::ASCENDING, OrderByNullType::NULLS_FIRST, wexpr->partitions[prt_idx]->Copy(),
			                    nullptr);
		} else {
			orders.emplace_back(OrderType::ASCENDING, OrderByNullType::NULLS_FIRST, wexpr->partitions[prt_idx]->Copy(),
			                    wexpr->partitions_stats[prt_idx]->Copy());
		}
	}
	for (const auto &order : wexpr->orders) {
		orders.push_back(order.Copy());
	}

	// fuse input and sort collection into one
	// (sorting columns are not decoded, and we need them later)
	ChunkCollection payload;
	payload.Fuse(input);
	payload.Fuse(over);
	auto payload_types = payload.Types();

	// initialise partitioning memory
	// to minimise copying, we fill up a chunk and then sink it.
	SelectionVector sel;
	DataChunk over_partition;
	DataChunk payload_partition;
	if (hashes) {
		sel.Initialize(STANDARD_VECTOR_SIZE);
		over_partition.Initialize(over.Types());
		payload_partition.Initialize(payload_types);
	}

	// initialize row layout for sorting
	RowLayout payload_layout;
	payload_layout.Initialize(payload_types);

	// initialize sorting states
	state.global_sort_state = make_unique<GlobalSortState>(state.buffer_manager, orders, payload_layout);
	auto &global_sort_state = *state.global_sort_state;
	LocalSortState local_sort_state;
	local_sort_state.Initialize(global_sort_state, state.buffer_manager);

	// sink collection chunks into row format
	const idx_t chunk_count = over.ChunkCount();
	for (idx_t i = 0; i < chunk_count; i++) {
		auto &over_chunk = *over.Chunks()[i];
		auto &payload_chunk = *payload.Chunks()[i];

		// Extract the hash partition, if any
		if (hashes) {
			auto &hash_chunk = *hashes->Chunks()[i];
			auto hash_size = hash_chunk.size();
			auto hash_data = FlatVector::GetData<hash_t>(hash_chunk.data[0]);
			idx_t bin_size = 0;
			for (idx_t i = 0; i < hash_size; ++i) {
				if ((hash_data[i] & hash_mask) == hash_bin) {
					sel.set_index(bin_size++, i);
				}
			}

			// Flush the partition chunks if we would overflow
			if (over_partition.size() + bin_size > STANDARD_VECTOR_SIZE) {
				local_sort_state.SinkChunk(over_partition, payload_partition);
				over_partition.Reset();
				payload_partition.Reset();
			}

			// Copy the data for each collection.
			if (bin_size) {
				over_partition.Append(over_chunk, false, &sel, bin_size);
				payload_partition.Append(payload_chunk, false, &sel, bin_size);
			}
		} else {
			local_sort_state.SinkChunk(over_chunk, payload_chunk);
		}
	}

	// Flush any ragged partition chunks
	if (over_partition.size() > 0) {
		local_sort_state.SinkChunk(over_partition, payload_partition);
		over_partition.Reset();
		payload_partition.Reset();
	}

	// If there are no hashes, release the input to save memory.
	if (!hashes) {
		over.Reset();
		input.Reset();
	}

	// add local state to global state, which sorts the data
	global_sort_state.AddLocalState(local_sort_state);
	// Prepare for merge phase (in this case we never have a merge phase, but this call is still needed)
	global_sort_state.PrepareMergePhase();
}
