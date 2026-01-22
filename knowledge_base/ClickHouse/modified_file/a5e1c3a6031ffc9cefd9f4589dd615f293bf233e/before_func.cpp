BlockIO InterpreterCheckQuery::execute()
{
	ASTCheckQuery & alter = typeid_cast<ASTCheckQuery &>(*query_ptr);
	String & table_name = alter.table;
	String database_name = alter.database.empty() ? context.getCurrentDatabase() : alter.database;

	StoragePtr table = context.getTable(database_name, table_name);

	auto distributed_table = typeid_cast<StorageDistributed *>(&*table);
	if (distributed_table != nullptr)
	{
		/// Для таблиц с движком Distributed запрос CHECK TABLE отправляет запрос DESCRIBE TABLE на все реплики.
		/// Проверяется идентичность структур (имена столбцов + типы столбцов + типы по-умолчанию + выражения
		/// по-умолчанию) таблиц, на котороые смотрит распределённая таблица.

		const auto settings = context.getSettings();

		BlockInputStreams streams = distributed_table->describe(context, settings);
		streams[0] = new UnionBlockInputStream<StreamUnionMode::ExtraInfo>(streams, nullptr, settings.max_threads);
		streams.resize(1);

		auto stream_ptr = dynamic_cast<IProfilingBlockInputStream *>(&*streams[0]);
		if (stream_ptr == nullptr)
			throw Exception("InterpreterCheckQuery: Internal error", ErrorCodes::LOGICAL_ERROR);
		auto & stream = *stream_ptr;

		/// Получить все данные от запросов DESCRIBE TABLE.

		TableDescriptions table_descriptions;

		while (true)
		{
			if (stream.isCancelled())
			{
				BlockIO res;
				res.in = new OneBlockInputStream(result);
				return res;
			}

			Block block = stream.read();
			if (!block)
				break;

			BlockExtraInfo info = stream.getBlockExtraInfo();
			if (!info.is_valid)
				throw Exception("Received invalid block extra info", ErrorCodes::INVALID_BLOCK_EXTRA_INFO);

			table_descriptions.emplace_back(block, info);
		}

		if (table_descriptions.empty())
			throw Exception("Received empty data", ErrorCodes::RECEIVED_EMPTY_DATA);

		/// Определить класс эквивалентности каждой структуры таблицы.

		std::sort(table_descriptions.begin(), table_descriptions.end());

		UInt32 structure_class = 0;

		auto it = table_descriptions.begin();
		it->structure_class = structure_class;

		auto prev = it;
		for (++it; it != table_descriptions.end(); ++it)
		{
			if (*prev < *it)
				++structure_class;
			it->structure_class = structure_class;
			prev = it;
		}

		/// Составить результат.

		ColumnPtr status_column = new ColumnUInt8;
		ColumnPtr host_name_column = new ColumnString;
		ColumnPtr host_address_column = new ColumnString;
		ColumnPtr port_column = new ColumnUInt16;
		ColumnPtr user_column = new ColumnString;
		ColumnPtr structure_class_column = new ColumnUInt32;
		ColumnPtr structure_column = new ColumnString;

		/// Это значение равно 1, если структура нигде не отлчиается, а 0 в противном случае.
		UInt8 status_value = (structure_class == 0) ? 1 : 0;

		for (const auto & desc : table_descriptions)
		{
			status_column->insert(static_cast<UInt64>(status_value));
			structure_class_column->insert(static_cast<UInt64>(desc.structure_class));
			host_name_column->insert(desc.extra_info.host);
			host_address_column->insert(desc.extra_info.resolved_address);
			port_column->insert(static_cast<UInt64>(desc.extra_info.port));
			user_column->insert(desc.extra_info.user);
			structure_column->insert(desc.names_with_types);
		}

		Block block;

		block.insert(ColumnWithTypeAndName(status_column, new DataTypeUInt8, "status"));
		block.insert(ColumnWithTypeAndName(host_name_column, new DataTypeString, "host_name"));
		block.insert(ColumnWithTypeAndName(host_address_column, new DataTypeString, "host_address"));
		block.insert(ColumnWithTypeAndName(port_column, new DataTypeUInt16, "port"));
		block.insert(ColumnWithTypeAndName(user_column, new DataTypeString, "user"));
		block.insert(ColumnWithTypeAndName(structure_class_column, new DataTypeUInt32, "structure_class"));
		block.insert(ColumnWithTypeAndName(structure_column, new DataTypeString, "structure"));

		BlockIO res;
		res.in = new OneBlockInputStream(block);
		res.in_sample = getSampleBlock();

		return res;
	}
	else
	{
		result = Block{{ new ColumnUInt8, new DataTypeUInt8, "result" }};
		result.getByPosition(0).column->insert(Field(UInt64(table->checkData())));

		BlockIO res;
		res.in = new OneBlockInputStream(result);
		res.in_sample = result.cloneEmpty();

		return res;
	}
}
