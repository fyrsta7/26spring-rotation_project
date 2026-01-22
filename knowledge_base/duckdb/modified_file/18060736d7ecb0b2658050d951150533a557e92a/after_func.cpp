WindowSegmentTree::WindowSegmentTree(AggregateObject aggr_p, const LogicalType &result_type_p, DataChunk *input,
                                     const ValidityMask &filter_mask_p, WindowAggregationMode mode_p)
    : aggr(std::move(aggr_p)), result_type(result_type_p), state(aggr.function.state_size()),
      statep(Value::POINTER(CastPointerToValue(state.data()))), frame(0, 0), statel(LogicalType::POINTER),
      statef(Value::POINTER(CastPointerToValue(state.data()))), flush_count(0), internal_nodes(0), input_ref(input),
      filter_mask(filter_mask_p), mode(mode_p) {
	statep.Flatten(STANDARD_VECTOR_SIZE);
	statef.SetVectorType(VectorType::FLAT_VECTOR); // Prevent conversion of results to constants

	if (input_ref && input_ref->ColumnCount() > 0) {
		inputs.Initialize(Allocator::DefaultAllocator(), input_ref->GetTypes());
		inputs.Reference(*input_ref);
		if (aggr.function.window && UseWindowAPI()) {
			// if we have a frame-by-frame method, share the single state
			AggregateInit();
		} else {
			//	In order to share the SV, we can't have any leaves that are already dictionaries.
			for (auto &leaf : inputs.data) {
				D_ASSERT(leaf.GetVectorType() == VectorType::FLAT_VECTOR);
			}
			//	The inputs share an SV so we can quickly pick out values
			//	TODO: Check after full vectorisation that this is still needed
			//	instad of just Slice/Reference.
			filter_sel.Initialize();
			//	What we slice to is not important now - we just want the SV to be shared.
			inputs.Slice(filter_sel, STANDARD_VECTOR_SIZE);
			if (aggr.function.combine && UseCombineAPI()) {
				ConstructTree();
			}
		}
	}
}
