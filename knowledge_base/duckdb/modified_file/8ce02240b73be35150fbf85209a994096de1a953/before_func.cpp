void PhysicalHashJoin::GetChunkInternal(ExecutionContext &context, DataChunk &chunk,
                                        PhysicalOperatorState *state_p) const {
	auto state = reinterpret_cast<PhysicalHashJoinState *>(state_p);
	auto &sink = (HashJoinGlobalState &)*sink_state;
	if (sink.hash_table->size() == 0 &&
	    (sink.hash_table->join_type == JoinType::INNER || sink.hash_table->join_type == JoinType::SEMI)) {
		// empty hash table with INNER or SEMI join means empty result set
		return;
	}
	do {
		ProbeHashTable(context, chunk, state);
		if (chunk.size() == 0) {
#if STANDARD_VECTOR_SIZE >= 128
			if (state->cached_chunk.size() > 0) {
				// finished probing but cached data remains, return cached chunk
				chunk.Reference(state->cached_chunk);
				state->cached_chunk.Reset();
			} else
#endif
			    if (IsRightOuterJoin(join_type)) {
				// check if we need to scan any unmatched tuples from the RHS for the full/right outer join
				sink.hash_table->ScanFullOuter(chunk, sink.ht_scan_state);
			}
			return;
		} else {
#if STANDARD_VECTOR_SIZE >= 128
			if (chunk.size() < 64) {
				// small chunk: add it to chunk cache and continue
				state->cached_chunk.Append(chunk);
				if (state->cached_chunk.size() >= (STANDARD_VECTOR_SIZE - 64)) {
					// chunk cache full: return it
					chunk.Reference(state->cached_chunk);
					state->cached_chunk.Reset();
					return;
				} else {
					// chunk cache not full: probe again
					chunk.Reset();
				}
			} else {
				return;
			}
#else
			return;
#endif
		}
	} while (true);
}
