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
			//	In order to share the SV, we can't have any inputs that are already dictionaries.
			for (column_t i = 0; i < inputs.ColumnCount(); i++) {
				auto &v = inputs.data[i];
				switch (v.GetVectorType()) {
				case VectorType::DICTIONARY_VECTOR:
				case VectorType::FSST_VECTOR:
					v.Flatten(input->size());
				default:
					break;
				}
			}
			//	The inputs share an SV so we can quickly pick out values
			filter_sel.Initialize();
			//	What we slice to is not important now - we just want the SV to be shared.
			inputs.Slice(filter_sel, STANDARD_VECTOR_SIZE);
			if (aggr.function.combine && UseCombineAPI()) {
				ConstructTree();
			}
		}
	}
}
