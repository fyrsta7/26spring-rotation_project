	SortedAggregateBindData(ClientContext &context, BoundAggregateExpression &expr)
	    : buffer_manager(BufferManager::GetBufferManager(context)), function(expr.function),
	      bind_info(std::move(expr.bind_info)), threshold(ClientConfig::GetConfig(context).ordered_aggregate_threshold),
	      external(ClientConfig::GetConfig(context).force_external) {
		auto &children = expr.children;
		arg_types.reserve(children.size());
		arg_funcs.reserve(children.size());
		for (const auto &child : children) {
			arg_types.emplace_back(child->return_type);
			ListSegmentFunctions funcs;
			GetSegmentDataFunctions(funcs, arg_types.back());
			arg_funcs.emplace_back(std::move(funcs));
		}
		auto &order_bys = *expr.order_bys;
		sort_types.reserve(order_bys.orders.size());
		sort_funcs.reserve(order_bys.orders.size());
		for (auto &order : order_bys.orders) {
			orders.emplace_back(order.Copy());
			sort_types.emplace_back(order.expression->return_type);
			ListSegmentFunctions funcs;
			GetSegmentDataFunctions(funcs, sort_types.back());
			sort_funcs.emplace_back(std::move(funcs));
		}
		sorted_on_args = (children.size() == order_bys.orders.size());
		for (size_t i = 0; sorted_on_args && i < children.size(); ++i) {
			sorted_on_args = children[i]->Equals(*order_bys.orders[i].expression);
		}
	}
