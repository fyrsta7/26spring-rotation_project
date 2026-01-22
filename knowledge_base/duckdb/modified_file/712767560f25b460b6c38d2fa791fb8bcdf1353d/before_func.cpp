void ExpressionExecutor::Visit(AggregateExpression &expr) {
	auto state =
	    reinterpret_cast<PhysicalAggregateOperatorState *>(this->state);
	if (!state) {
		throw NotImplementedException("Aggregate node without aggregate state");
	}
	if (state->aggregates.size() == 0) {
		if (state->aggregate_chunk.column_count &&
		    state->aggregate_chunk.data[expr.index].count) {
			vector.Reference(state->aggregate_chunk.data[expr.index]);
		} else {
			// the subquery scanned no rows, therefore the aggr is empty. return
			// something reasonable depending on aggr type.
			Value val;
			if (expr.type == ExpressionType::AGGREGATE_COUNT ||
			    expr.type == ExpressionType::AGGREGATE_COUNT_STAR) {
				val = Value(0); // ZERO
			} else {
				val = Value(); // NULL
			}
			Vector v(val);
			v.Move(vector);
		}
	} else {
		Vector v(state->aggregates[expr.index]);
		v.Move(vector);
	}
}
