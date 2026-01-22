unique_ptr<PhysicalOperator> PhysicalPlanGenerator::CreatePlan(LogicalLimit &op) {
	D_ASSERT(op.children.size() == 1);

	auto plan = CreatePlan(*op.children[0]);

	if (plan->type == PhysicalOperatorType::ORDER_BY && op.limit_val.Type() == LimitNodeType::CONSTANT_VALUE &&
	    op.offset_val.Type() != LimitNodeType::EXPRESSION_VALUE) {
		auto &order_by = plan->Cast<PhysicalOrder>();
		// Can not use TopN operator if PhysicalOrder uses projections
		bool omit_projection = true;
		for (idx_t i = 0; i < order_by.projections.size(); i++) {
			if (order_by.projections[i] == i) {
				continue;
			}
			omit_projection = false;
			break;
		}
		if (order_by.projections.empty() || omit_projection)
		{
			idx_t offset_val = 0;
			if (op.offset_val.Type() == LimitNodeType::CONSTANT_VALUE) {
				offset_val = op.offset_val.GetConstantValue();
			}
			auto top_n = make_uniq<PhysicalTopN>(order_by.children[0]->types, std::move(order_by.orders), op.limit_val.GetConstantValue(),
			                                     offset_val, op.estimated_cardinality);
			top_n->children.push_back(std::move(order_by.children[0]));
			return std::move(top_n);
		}
	}

	unique_ptr<PhysicalOperator> limit;
	switch (op.limit_val.Type()) {
	case LimitNodeType::EXPRESSION_PERCENTAGE:
	case LimitNodeType::CONSTANT_PERCENTAGE:
		limit = make_uniq<PhysicalLimitPercent>(op.types, std::move(op.limit_val), std::move(op.offset_val),
		                                        op.estimated_cardinality);
		break;
	default:
		if (!PreserveInsertionOrder(*plan)) {
			// use parallel streaming limit if insertion order is not important
			limit = make_uniq<PhysicalStreamingLimit>(op.types, std::move(op.limit_val), std::move(op.offset_val),
			                                          op.estimated_cardinality, true);
		} else {
			// maintaining insertion order is important
			if (UseBatchIndex(*plan) && UseBatchLimit(op.limit_val, op.offset_val)) {
				// source supports batch index: use parallel batch limit
				limit = make_uniq<PhysicalLimit>(op.types, std::move(op.limit_val), std::move(op.offset_val),
				                                 op.estimated_cardinality);
			} else {
				// source does not support batch index: use a non-parallel streaming limit
				limit = make_uniq<PhysicalStreamingLimit>(op.types, std::move(op.limit_val), std::move(op.offset_val),
				                                          op.estimated_cardinality, false);
			}
		}
		break;
	}

	limit->children.push_back(std::move(plan));
	return limit;
}
