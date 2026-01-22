unique_ptr<LogicalOperator> FilterPushdown::PushdownAggregate(unique_ptr<LogicalOperator> op) {
	D_ASSERT(op->type == LogicalOperatorType::LOGICAL_AGGREGATE_AND_GROUP_BY);
	auto &aggr = op->Cast<LogicalAggregate>();

	// pushdown into AGGREGATE and GROUP BY
	// we cannot push expressions that refer to the aggregate
	FilterPushdown child_pushdown(optimizer, convert_mark_joins);
	for (idx_t i = 0; i < filters.size(); i++) {
		auto &f = *filters[i];
		if (f.bindings.find(aggr.aggregate_index) != f.bindings.end()) {
			// filter on aggregate: cannot pushdown
			continue;
		}
		if (f.bindings.find(aggr.groupings_index) != f.bindings.end()) {
			// filter on GROUPINGS function: cannot pushdown
			continue;
		}
		// no aggregate! we are filtering on a group
		// we can only push this down if the filter is in all grouping sets
		vector<ColumnBinding> bindings;
		ExtractFilterBindings(*f.filter, bindings);

		bool can_pushdown_filter = true;
		if (aggr.grouping_sets.empty()) {
			// empty grouping set - we cannot pushdown the filter
			can_pushdown_filter = false;
		}
		if (bindings.empty()) {
			// we can never push down empty grouping sets
			continue;
		}
		for (auto &grp : aggr.grouping_sets) {
			// check for each of the grouping sets if they contain all groups
			for (auto &binding : bindings) {
				if (grp.find(binding.column_index) == grp.end()) {
					can_pushdown_filter = false;
					break;
				}
			}
			if (!can_pushdown_filter) {
				break;
			}
		}
		if (!can_pushdown_filter) {
			continue;
		}
		// no aggregate! we can push this down
		// rewrite any group bindings within the filter
		f.filter = ReplaceGroupBindings(aggr, std::move(f.filter));
		// add the filter to the child node
		if (child_pushdown.AddFilter(std::move(f.filter)) == FilterResult::UNSATISFIABLE) {
			// filter statically evaluates to false, strip tree
			return make_uniq<LogicalEmptyResult>(std::move(op));
		}
		// erase the filter from here
		filters.erase_at(i);
		i--;
	}
	child_pushdown.GenerateFilters();

	op->children[0] = child_pushdown.Rewrite(std::move(op->children[0]));
	return FinishPushdown(std::move(op));
}
