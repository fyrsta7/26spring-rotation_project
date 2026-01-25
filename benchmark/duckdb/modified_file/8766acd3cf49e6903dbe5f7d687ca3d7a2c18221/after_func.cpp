Index::Index(AttachedDatabase &db, IndexType type, TableIOManager &table_io_manager,
             const vector<column_t> &column_ids_p, const vector<unique_ptr<Expression>> &unbound_expressions,
             IndexConstraintType constraint_type_p)

    : type(type), table_io_manager(table_io_manager), column_ids(column_ids_p), constraint_type(constraint_type_p),
      db(db), buffer_manager(BufferManager::GetBufferManager(db)) {

	for (auto &expr : unbound_expressions) {
		types.push_back(expr->return_type.InternalType());
		logical_types.push_back(expr->return_type);
		auto unbound_expression = expr->Copy();
		bound_expressions.push_back(BindExpression(unbound_expression->Copy()));
		this->unbound_expressions.emplace_back(std::move(unbound_expression));
	}
	for (auto &bound_expr : bound_expressions) {
		executor.AddExpression(*bound_expr);
	}

	// create the column id set
	column_id_set.insert(column_ids.begin(), column_ids.end());
}