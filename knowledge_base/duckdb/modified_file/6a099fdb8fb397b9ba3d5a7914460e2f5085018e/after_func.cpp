void BaseAppender::AppendDataChunk(DataChunk &chunk_p) {
	auto chunk_types = chunk_p.GetTypes();

	// Early-out, if types match.
	if (chunk_types == types) {
		collection->Append(chunk_p);
		if (collection->Count() >= flush_count) {
			Flush();
		}
		return;
	}

	auto count = chunk_p.ColumnCount();
	if (count != types.size()) {
		throw InvalidInputException("incorrect column count in AppendDataChunk, expected %d, got %d", types.size(),
		                            count);
	}

	// We try to cast the chunk.
	auto size = chunk_p.size();
	DataChunk cast_chunk;
	cast_chunk.Initialize(allocator, types);
	cast_chunk.SetCardinality(size);

	for (idx_t i = 0; i < count; i++) {
		if (chunk_p.data[i].GetType() == types[i]) {
			cast_chunk.data[i].Reference(chunk_p.data[i]);
			continue;
		}

		string error_msg;
		auto success = VectorOperations::DefaultTryCast(chunk_p.data[i], cast_chunk.data[i], size, &error_msg);
		if (!success) {
			throw InvalidInputException("type mismatch in AppendDataChunk, expected %s, got %s for column %d",
			                            types[i].ToString(), chunk_p.data[i].GetType().ToString(), i);
		}
	}

	collection->Append(cast_chunk);
	if (collection->Count() >= flush_count) {
		Flush();
	}
}
