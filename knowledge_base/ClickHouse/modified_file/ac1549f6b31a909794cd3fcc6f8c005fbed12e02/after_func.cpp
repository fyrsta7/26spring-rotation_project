        /// Physical columns go first and then some virtual columns follow
        /// TODO: is there a better way to account for virtual columns that were filled by previous readers?
//        size_t physical_columns_count = read_result.columns.size() - read_result.extra_columns_filled.size();
//        Columns physical_columns(read_result.columns.begin(), read_result.columns.begin() + physical_columns_count);

        bool should_evaluate_missing_defaults;
        merge_tree_reader->fillMissingColumns(physical_columns, should_evaluate_missing_defaults,
                                                read_result.num_rows);

        /// If some columns absent in part, then evaluate default values
        if (should_evaluate_missing_defaults)
            // TODO: must pass proper block here, not block_before_prewhere!
            merge_tree_reader->evaluateMissingDefaults(read_result.block_before_prewhere, physical_columns);

        /// If result not empty, then apply on-fly alter conversions if any required
        merge_tree_reader->performRequiredConversions(physical_columns);

/*
        for (const auto & column_name : non_const_virtual_column_names)
        {
            if (column_name == "_part_offset")
            {
                // TODO: properly fill _part_offset!
                physical_columns.emplace_back(ColumnUInt64::create(read_result.num_rows));
            }
        }
//*/
//        for (size_t i = 0; i < physical_columns.size(); ++i)
//            read_result.columns[i] = std::move(physical_columns[i]);

//        read_result.checkInternalConsistency();
    }
}


void MergeTreeRangeReader::executePrewhereActionsAndFilterColumns(ReadResult & result) const
{
    result.checkInternalConsistency();

    if (!prewhere_info)
        return;

    const auto & header = merge_tree_reader->getColumns();
    size_t num_columns = header.size();

    /// Check that we have columns from previous steps and newly read required columns
    if (result.columns.size() < num_columns + result.extra_columns_filled.size())
        throw Exception(ErrorCodes::LOGICAL_ERROR,
                        "Invalid number of columns passed to MergeTreeRangeReader. Expected {}, got {}",
                        num_columns, result.columns.size());

    /// This filter has the size of total_rows_per granule. It is applied after reading contiguous chunks from
    /// the start of each granule.
//    ColumnPtr combined_filter;
    /// Filter computed at the current step. Its size is equal to num_rows which is <= total_rows_per_granule
    ColumnPtr current_step_filter;
    size_t prewhere_column_pos;

    {
        /// Restore block from columns list.
        Block block;
        size_t pos = 0;

        if (prev_reader)
        {
            for (const auto & col : prev_reader->getSampleBlock())
            {
                block.insert({result.columns[pos], col.type, col.name});
                ++pos;
            }
        }

        for (auto name_and_type = header.begin(); name_and_type != header.end() && pos < result.columns.size(); ++pos, ++name_and_type)
            block.insert({result.columns[pos], name_and_type->type, name_and_type->name});


    /*// HACK!! fix it
        if (getSampleBlock().has("_part_offset"))
        {
            const auto & col = getSampleBlock().getByName("_part_offset");
            block.insert({result.columns.back(), col.type, col.name});
        }
/////////////*/

        {
            /// Columns might be projected out. We need to store them here so that default columns can be evaluated later.
            Block block_before_prewhere = block;

            if (prewhere_info->actions)
                prewhere_info->actions->execute(block);

            result.block_before_prewhere.clear();
            for (auto & col : block_before_prewhere)
            {
                /// Exclude columns that are present in the result block to avoid storing them and filtering twice
                if (block.has(col.name))
                    continue;
                result.block_before_prewhere.insert(col);
            }
        }

        prewhere_column_pos = block.getPositionByName(prewhere_info->column_name);

        result.columns.clear();
        result.columns.reserve(block.columns());
        for (auto & col : block)
            result.columns.emplace_back(std::move(col.column));

        current_step_filter = result.columns[prewhere_column_pos];
//        combined_filter = current_step_filter;
    }

    if (prewhere_info->remove_column)
