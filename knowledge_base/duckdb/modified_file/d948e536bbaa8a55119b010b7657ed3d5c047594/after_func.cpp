OperatorResultType PhysicalStreamingWindow::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                    GlobalOperatorState &gstate_p, OperatorState &state_p) const {
	auto &gstate = (StreamingWindowGlobalState &)gstate_p;
	auto &state = (StreamingWindowState &)state_p;
	if (!state.initialized) {
		auto &allocator = Allocator::Get(context.client);
		state.Initialize(allocator, input, select_list);
	}
	// Put payload columns in place
	for (idx_t col_idx = 0; col_idx < input.data.size(); col_idx++) {
		chunk.data[col_idx].Reference(input.data[col_idx]);
	}
	// Compute window function
	const idx_t count = input.size();
	for (idx_t expr_idx = 0; expr_idx < select_list.size(); expr_idx++) {
		idx_t col_idx = input.data.size() + expr_idx;
		auto &expr = *select_list[expr_idx];
		auto &result = chunk.data[col_idx];
		switch (expr.GetExpressionType()) {
		case ExpressionType::WINDOW_AGGREGATE: {
			//	Establish the aggregation environment
			auto &wexpr = (BoundWindowExpression &)expr;
			auto &aggregate = *wexpr.aggregate;
			auto &statev = state.statev;
			state.state_ptr = state.aggregate_states[expr_idx].data();
			AggregateInputData aggr_input_data(wexpr.bind_info.get());

			// Check for COUNT(*)
			if (wexpr.children.empty()) {
				D_ASSERT(GetTypeIdSize(result.GetType().InternalType()) == sizeof(int64_t));
				auto data = FlatVector::GetData<int64_t>(result);
				for (idx_t i = 0; i < input.size(); ++i) {
					data[i] = gstate.row_number + i;
				}
				break;
			}

			// Compute the arguments
			auto &allocator = Allocator::Get(context.client);
			ExpressionExecutor executor(allocator);
			vector<LogicalType> payload_types;
			for (auto &child : wexpr.children) {
				payload_types.push_back(child->return_type);
				executor.AddExpression(*child);
			}

			DataChunk payload;
			payload.Initialize(executor.allocator, payload_types);
			executor.Execute(input, payload);

			// Iterate through them using a single SV
			payload.Flatten();
			DataChunk row;
			row.Initialize(allocator, payload_types);
			sel_t s = 0;
			SelectionVector sel(&s);
			row.Slice(sel, 1);
			for (size_t col_idx = 0; col_idx < payload.ColumnCount(); ++col_idx) {
				DictionaryVector::Child(row.data[col_idx]).Reference(payload.data[col_idx]);
			}

			// Update the state and finalize it one row at a time.
			for (idx_t i = 0; i < input.size(); ++i) {
				sel.set_index(0, i);
				aggregate.update(row.data.data(), aggr_input_data, row.ColumnCount(), statev, 1);
				aggregate.finalize(statev, aggr_input_data, result, 1, i);
			}
			break;
		}
		case ExpressionType::WINDOW_FIRST_VALUE:
		case ExpressionType::WINDOW_PERCENT_RANK:
		case ExpressionType::WINDOW_RANK:
		case ExpressionType::WINDOW_RANK_DENSE: {
			// Reference constant vector
			chunk.data[col_idx].Reference(*state.const_vectors[expr_idx]);
			break;
		}
		case ExpressionType::WINDOW_ROW_NUMBER: {
			// Set row numbers
			auto rdata = FlatVector::GetData<int64_t>(chunk.data[col_idx]);
			for (idx_t i = 0; i < count; i++) {
				rdata[i] = gstate.row_number + i;
			}
			break;
		}
		default:
			throw NotImplementedException("%s for StreamingWindow", ExpressionTypeToString(expr.GetExpressionType()));
		}
	}
	gstate.row_number += count;
	chunk.SetCardinality(count);
	return OperatorResultType::NEED_MORE_INPUT;
}
