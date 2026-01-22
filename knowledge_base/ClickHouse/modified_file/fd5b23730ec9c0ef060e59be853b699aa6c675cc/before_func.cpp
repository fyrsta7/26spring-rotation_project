BlockInputStreams StorageMerge::read(
    const Names & column_names,
    const SelectQueryInfo & query_info,
    const Context & context,
    QueryProcessingStage::Enum processed_stage,
    const size_t max_block_size,
    unsigned num_streams)
{
    BlockInputStreams res;

    bool has_table_virtual_column = false;
    Names real_column_names;
    real_column_names.reserve(column_names.size());

    for (const auto & column_name : column_names)
    {
        if (column_name == "_table" && isVirtualColumn(column_name))
            has_table_virtual_column = true;
        else
            real_column_names.push_back(column_name);
    }

    /** Just in case, turn off optimization "transfer to PREWHERE",
      * since there is no certainty that it works when one of table is MergeTree and other is not.
      */
    Context modified_context = context;
    modified_context.getSettingsRef().optimize_move_to_prewhere = false;

    /// What will be result structure depending on query processed stage in source tables?
    Block header = getQueryHeader(column_names, query_info, context, processed_stage);

    /** First we make list of selected tables to find out its size.
      * This is necessary to correctly pass the recommended number of threads to each table.
      */
    StorageListWithLocks selected_tables = getSelectedTables(
        query_info.query, has_table_virtual_column, true, context.getCurrentQueryId());

    if (selected_tables.empty())
        /// FIXME: do we support sampling in this case?
        return createSourceStreams(
            query_info, processed_stage, max_block_size, header, {}, {}, real_column_names, modified_context, 0, has_table_virtual_column);

    size_t tables_count = selected_tables.size();
    Float64 num_streams_multiplier = std::min(unsigned(tables_count), std::max(1U, unsigned(context.getSettingsRef().max_streams_multiplier_for_merge_tables)));
    num_streams *= num_streams_multiplier;
    size_t remaining_streams = num_streams;

    for (auto it = selected_tables.begin(); it != selected_tables.end(); ++it)
    {
        size_t current_need_streams = tables_count >= num_streams ? 1 : (num_streams / tables_count);
        size_t current_streams = std::min(current_need_streams, remaining_streams);
        remaining_streams -= current_streams;
        current_streams = std::max(size_t(1), current_streams);

        StoragePtr storage = it->first;
        TableStructureReadLockHolder struct_lock = it->second;

        /// If sampling requested, then check that table supports it.
        if (query_info.query->as<ASTSelectQuery>()->sample_size() && !storage->supportsSampling())
            throw Exception("Illegal SAMPLE: table doesn't support sampling", ErrorCodes::SAMPLING_NOT_SUPPORTED);

        BlockInputStreams source_streams;

        if (current_streams)
        {
            source_streams = createSourceStreams(
                query_info, processed_stage, max_block_size, header, storage,
                struct_lock, real_column_names, modified_context, current_streams, has_table_virtual_column);
        }
        else
        {
            source_streams.emplace_back(std::make_shared<LazyBlockInputStream>(
                header, [=]() mutable -> BlockInputStreamPtr
                {
                    BlockInputStreams streams = createSourceStreams(query_info, processed_stage, max_block_size,
                                                                    header, storage, struct_lock, real_column_names,
                                                                    modified_context, current_streams, has_table_virtual_column, true);

                    if (!streams.empty() && streams.size() != 1)
                        throw Exception("LogicalError: the lazy stream size must to be one or empty.", ErrorCodes::LOGICAL_ERROR);

                    return streams.empty() ? std::make_shared<NullBlockInputStream>(header) : streams[0];
                }));
        }

        res.insert(res.end(), source_streams.begin(), source_streams.end());
    }

    if (res.empty())
        return res;
