BlockInputStreams MergeTreeDataSelectExecutor::read(
	const Names & column_names_to_return,
	ASTPtr query,
	const Settings & settings,
	QueryProcessingStage::Enum & processed_stage,
	size_t max_block_size,
	unsigned threads,
	size_t * part_index)
{
	size_t part_index_var = 0;
	if (!part_index)
		part_index = &part_index_var;

	MergeTreeData::DataPartsVector parts = data.getDataPartsVector();

	/// Если в запросе есть ограничения на виртуальный столбец _part, выберем только подходящие под него куски.
	Names virt_column_names, real_column_names;
	for (const String & name : column_names_to_return)
		if (name != "_part" &&
			name != "_part_index")
			real_column_names.push_back(name);
		else
			virt_column_names.push_back(name);

	/// Если в запросе только виртуальные столбцы, надо запросить хотя бы один любой другой.
	if (real_column_names.size() == 0)
		real_column_names.push_back(ExpressionActions::getSmallestColumn(data.getColumnsList()));

	Block virtual_columns_block = getBlockWithVirtualColumns(parts);

	/// Если запрошен хотя бы один виртуальный столбец, пробуем индексировать
	if (!virt_column_names.empty())
		VirtualColumnUtils::filterBlockWithQuery(query->clone(), virtual_columns_block, data.context);

	std::multiset<String> values = VirtualColumnUtils::extractSingleValueFromBlock<String>(virtual_columns_block, "_part");

	data.check(real_column_names);
	processed_stage = QueryProcessingStage::FetchColumns;

	PKCondition key_condition(query, data.context, data.getColumnsList(), data.getSortDescription());
	PKCondition date_condition(query, data.context, data.getColumnsList(), SortDescription(1, SortColumnDescription(data.date_column_name, 1)));

	/// Выберем куски, в которых могут быть данные, удовлетворяющие date_condition, и которые подходят под условие на _part.
	{
		auto prev_parts = parts;
		parts.clear();

		for (const auto & part : prev_parts)
		{
			if (values.find(part->name) == values.end())
				continue;

			Field left = static_cast<UInt64>(part->left_date);
			Field right = static_cast<UInt64>(part->right_date);

			if (!date_condition.mayBeTrueInRange(&left, &right))
				continue;

			parts.push_back(part);
		}
	}

	/// Семплирование.
	Names column_names_to_read = real_column_names;
	UInt64 sampling_column_value_limit = 0;
	typedef Poco::SharedPtr<ASTFunction> ASTFunctionPtr;
	ASTFunctionPtr filter_function;
	ExpressionActionsPtr filter_expression;

	ASTSelectQuery & select = *typeid_cast<ASTSelectQuery*>(&*query);
	if (settings.optimize_move_to_prewhere)
		if (select.where_expression && !select.prewhere_expression)
			MergeTreeWhereOptimizer{select, data, column_names_to_return, log};

	if (select.sample_size)
	{
		double size = apply_visitor(FieldVisitorConvertToNumber<double>(),
			typeid_cast<ASTLiteral&>(*select.sample_size).value);

		if (size < 0)
			throw Exception("Negative sample size", ErrorCodes::ARGUMENT_OUT_OF_BOUND);

		if (size > 1)
		{
			size_t requested_count = apply_visitor(FieldVisitorConvertToNumber<UInt64>(), typeid_cast<ASTLiteral&>(*select.sample_size).value);

			/// Узнаем, сколько строк мы бы прочли без семплирования.
			LOG_DEBUG(log, "Preliminary index scan with condition: " << key_condition.toString());
			size_t total_count = 0;
			for (size_t i = 0; i < parts.size(); ++i)
			{
				MergeTreeData::DataPartPtr & part = parts[i];
				MarkRanges ranges = markRangesFromPkRange(part->index, key_condition);

				for (size_t j = 0; j < ranges.size(); ++j)
					total_count += ranges[j].end - ranges[j].begin;
			}
			total_count *= data.index_granularity;

			size = std::min(1., static_cast<double>(requested_count) / total_count);

			LOG_DEBUG(log, "Selected relative sample size: " << size);
		}

		UInt64 sampling_column_max = 0;
		DataTypePtr type = data.getPrimaryExpression()->getSampleBlock().getByName(data.sampling_expression->getColumnName()).type;

		if (type->getName() == "UInt64")
			sampling_column_max = std::numeric_limits<UInt64>::max();
		else if (type->getName() == "UInt32")
			sampling_column_max = std::numeric_limits<UInt32>::max();
		else if (type->getName() == "UInt16")
			sampling_column_max = std::numeric_limits<UInt16>::max();
		else if (type->getName() == "UInt8")
			sampling_column_max = std::numeric_limits<UInt8>::max();
		else
			throw Exception("Invalid sampling column type in storage parameters: " + type->getName() + ". Must be unsigned integer type.", ErrorCodes::ILLEGAL_TYPE_OF_COLUMN_FOR_FILTER);

		/// Добавим условие, чтобы отсечь еще что-нибудь при повторном просмотре индекса.
		sampling_column_value_limit = static_cast<UInt64>(size * sampling_column_max);
		if (!key_condition.addCondition(data.sampling_expression->getColumnName(),
			Range::createRightBounded(sampling_column_value_limit, true)))
			throw Exception("Sampling column not in primary key", ErrorCodes::ILLEGAL_COLUMN);

		/// Выражение для фильтрации: sampling_expression <= sampling_column_value_limit

		ASTPtr filter_function_args = new ASTExpressionList;
		filter_function_args->children.push_back(data.sampling_expression);
		filter_function_args->children.push_back(new ASTLiteral(StringRange(), sampling_column_value_limit));

		filter_function = new ASTFunction;
		filter_function->name = "lessOrEquals";
		filter_function->arguments = filter_function_args;
		filter_function->children.push_back(filter_function->arguments);

		filter_expression = ExpressionAnalyzer(filter_function, data.context, data.getColumnsList()).getActions(false);

		/// Добавим столбцы, нужные для sampling_expression.
		std::vector<String> add_columns = filter_expression->getRequiredColumns();
		column_names_to_read.insert(column_names_to_read.end(), add_columns.begin(), add_columns.end());
		std::sort(column_names_to_read.begin(), column_names_to_read.end());
		column_names_to_read.erase(std::unique(column_names_to_read.begin(), column_names_to_read.end()), column_names_to_read.end());
	}

	LOG_DEBUG(log, "Key condition: " << key_condition.toString());
	LOG_DEBUG(log, "Date condition: " << date_condition.toString());

	/// PREWHERE
	ExpressionActionsPtr prewhere_actions;
	String prewhere_column;
	if (select.prewhere_expression)
	{
		ExpressionAnalyzer analyzer(select.prewhere_expression, data.context, data.getColumnsList());
		prewhere_actions = analyzer.getActions(false);
		prewhere_column = select.prewhere_expression->getColumnName();
		/// TODO: Чтобы работали подзапросы в PREWHERE, можно тут сохранить analyzer.getSetsWithSubqueries(), а потом их выполнить.
	}

	RangesInDataParts parts_with_ranges;

	/// Найдем, какой диапазон читать из каждого куска.
	size_t sum_marks = 0;
	size_t sum_ranges = 0;
	for (size_t i = 0; i < parts.size(); ++i)
	{
		MergeTreeData::DataPartPtr & part = parts[i];
		RangesInDataPart ranges(part, (*part_index)++);
		ranges.ranges = markRangesFromPkRange(part->index, key_condition);

		if (!ranges.ranges.empty())
		{
			parts_with_ranges.push_back(ranges);

			sum_ranges += ranges.ranges.size();
			for (size_t j = 0; j < ranges.ranges.size(); ++j)
			{
				sum_marks += ranges.ranges[j].end - ranges.ranges[j].begin;
			}
		}
	}

	LOG_DEBUG(log, "Selected " << parts.size() << " parts by date, " << parts_with_ranges.size() << " parts by key, "
			  << sum_marks << " marks to read from " << sum_ranges << " ranges");

	BlockInputStreams res;

	if (select.final)
	{
		/// Добавим столбцы, нужные для вычисления первичного ключа и знака.
		std::vector<String> add_columns = data.getPrimaryExpression()->getRequiredColumns();
		column_names_to_read.insert(column_names_to_read.end(), add_columns.begin(), add_columns.end());
		column_names_to_read.push_back(data.sign_column);
		std::sort(column_names_to_read.begin(), column_names_to_read.end());
		column_names_to_read.erase(std::unique(column_names_to_read.begin(), column_names_to_read.end()), column_names_to_read.end());

		res = spreadMarkRangesAmongThreadsFinal(
			parts_with_ranges,
			threads,
			column_names_to_read,
			max_block_size,
			settings.use_uncompressed_cache,
			prewhere_actions,
			prewhere_column,
			virt_column_names);
	}
	else
	{
		res = spreadMarkRangesAmongThreads(
			parts_with_ranges,
			threads,
			column_names_to_read,
			max_block_size,
			settings.use_uncompressed_cache,
			prewhere_actions,
			prewhere_column,
			virt_column_names);
	}

	if (select.sample_size)
	{
		for (size_t i = 0; i < res.size(); ++i)
		{
			BlockInputStreamPtr original_stream = res[i];
			BlockInputStreamPtr expression_stream = new ExpressionBlockInputStream(original_stream, filter_expression);
			BlockInputStreamPtr filter_stream = new FilterBlockInputStream(expression_stream, filter_function->getColumnName());
			res[i] = filter_stream;
		}
	}

	return res;
}
