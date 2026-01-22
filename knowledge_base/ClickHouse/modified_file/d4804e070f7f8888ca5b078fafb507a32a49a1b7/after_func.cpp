            }

            executeWithFill(pipeline);

            /** We must do projection after DISTINCT because projection may remove some columns.
              */
            executeProjection(pipeline, expressions.final_projection);

            /** Extremes are calculated before LIMIT, but after LIMIT BY. This is Ok.
              */
            executeExtremes(pipeline);

            if (!(pipeline_with_processors && has_prelimit))  /// Limit is no longer needed if there is prelimit.
                executeLimit(pipeline);

            executeOffset(pipeline);
        }
    }

    if (query_analyzer->hasGlobalSubqueries() && !subqueries_for_sets.empty())
        executeSubqueriesInSetsAndJoins(pipeline, subqueries_for_sets);
}

template <typename TPipeline>
void InterpreterSelectQuery::executeFetchColumns(
        QueryProcessingStage::Enum processing_stage, TPipeline & pipeline,
        const PrewhereInfoPtr & prewhere_info, const Names & columns_to_remove_after_prewhere,
        QueryPipeline & save_context_and_storage)
{
    constexpr bool pipeline_with_processors = std::is_same<TPipeline, QueryPipeline>::value;

    auto & query = getSelectQuery();
    const Settings & settings = context->getSettingsRef();

    /// Optimization for trivial query like SELECT count() FROM table.
    auto check_trivial_count_query = [&]() -> std::optional<AggregateDescription>
    {
        if (!settings.optimize_trivial_count_query || !syntax_analyzer_result->maybe_optimize_trivial_count || !storage
            || query.sampleSize() || query.sampleOffset() || query.final() || query.prewhere() || query.where() || query.groupBy()
            || !query_analyzer->hasAggregation() || processing_stage != QueryProcessingStage::FetchColumns)
            return {};

        const AggregateDescriptions & aggregates = query_analyzer->aggregates();

        if (aggregates.size() != 1)
            return {};

        const AggregateDescription & desc = aggregates[0];
        if (typeid_cast<AggregateFunctionCount *>(desc.function.get()))
            return desc;

        return {};
    };

    if (auto desc = check_trivial_count_query())
    {
        auto func = desc->function;
        std::optional<UInt64> num_rows = storage->totalRows();
        if (num_rows)
        {
            AggregateFunctionCount & agg_count = static_cast<AggregateFunctionCount &>(*func);

            /// We will process it up to "WithMergeableState".
            std::vector<char> state(agg_count.sizeOfData());
            AggregateDataPtr place = state.data();

            agg_count.create(place);
            SCOPE_EXIT(agg_count.destroy(place));

            agg_count.set(place, *num_rows);

            auto column = ColumnAggregateFunction::create(func);
            column->insertFrom(place);

            auto header = analysis_result.before_aggregation->getSampleBlock();
            size_t arguments_size = desc->argument_names.size();
            DataTypes argument_types(arguments_size);
            for (size_t j = 0; j < arguments_size; ++j)
                argument_types[j] = header.getByName(desc->argument_names[j]).type;

            Block block_with_count{
                {std::move(column), std::make_shared<DataTypeAggregateFunction>(func, argument_types, desc->parameters), desc->column_name}};

            auto istream = std::make_shared<OneBlockInputStream>(block_with_count);
            if constexpr (pipeline_with_processors)
                pipeline.init(Pipe(std::make_shared<SourceFromInputStream>(istream)));
            else
                pipeline.streams.emplace_back(istream);
            from_stage = QueryProcessingStage::WithMergeableState;
            analysis_result.first_stage = false;
            return;
        }
    }

    /// Actions to calculate ALIAS if required.
    ExpressionActionsPtr alias_actions;

    if (storage)
    {
        /// Append columns from the table filter to required
        auto row_policy_filter = context->getRowPolicyCondition(table_id.getDatabaseName(), table_id.getTableName(), RowPolicy::SELECT_FILTER);
        if (row_policy_filter)
        {
            auto initial_required_columns = required_columns;
            ExpressionActionsPtr actions;
            generateFilterActions(actions, row_policy_filter, initial_required_columns);
            auto required_columns_from_filter = actions->getRequiredColumns();

            for (const auto & column : required_columns_from_filter)
            {
                if (required_columns.end() == std::find(required_columns.begin(), required_columns.end(), column))
                    required_columns.push_back(column);
            }
        }

        /// Detect, if ALIAS columns are required for query execution
        auto alias_columns_required = false;
        const ColumnsDescription & storage_columns = storage->getColumns();
        for (const auto & column_name : required_columns)
        {
            auto column_default = storage_columns.getDefault(column_name);
            if (column_default && column_default->kind == ColumnDefaultKind::Alias)
            {
                alias_columns_required = true;
                break;
            }
        }

        /// There are multiple sources of required columns:
        ///  - raw required columns,
        ///  - columns deduced from ALIAS columns,
        ///  - raw required columns from PREWHERE,
        ///  - columns deduced from ALIAS columns from PREWHERE.
        /// PREWHERE is a special case, since we need to resolve it and pass directly to `IStorage::read()`
        /// before any other executions.
        if (alias_columns_required)
        {
            NameSet required_columns_from_prewhere; /// Set of all (including ALIAS) required columns for PREWHERE
            NameSet required_aliases_from_prewhere; /// Set of ALIAS required columns for PREWHERE

            if (prewhere_info)
            {
                /// Get some columns directly from PREWHERE expression actions
                auto prewhere_required_columns = prewhere_info->prewhere_actions->getRequiredColumns();
                required_columns_from_prewhere.insert(prewhere_required_columns.begin(), prewhere_required_columns.end());
            }

            /// Expression, that contains all raw required columns
            ASTPtr required_columns_all_expr = std::make_shared<ASTExpressionList>();

            /// Expression, that contains raw required columns for PREWHERE
            ASTPtr required_columns_from_prewhere_expr = std::make_shared<ASTExpressionList>();

            /// Sort out already known required columns between expressions,
            /// also populate `required_aliases_from_prewhere`.
            for (const auto & column : required_columns)
            {
                ASTPtr column_expr;
                const auto column_default = storage_columns.getDefault(column);
                bool is_alias = column_default && column_default->kind == ColumnDefaultKind::Alias;
                if (is_alias)
                {
                    auto column_decl = storage_columns.get(column);
                    /// TODO: can make CAST only if the type is different (but requires SyntaxAnalyzer).
                    auto cast_column_default = addTypeConversionToAST(column_default->expression->clone(), column_decl.type->getName());
                    column_expr = setAlias(cast_column_default->clone(), column);
                }
                else
                    column_expr = std::make_shared<ASTIdentifier>(column);

                if (required_columns_from_prewhere.count(column))
                {
                    required_columns_from_prewhere_expr->children.emplace_back(std::move(column_expr));

                    if (is_alias)
                        required_aliases_from_prewhere.insert(column);
                }
                else
                    required_columns_all_expr->children.emplace_back(std::move(column_expr));
            }

            /// Columns, which we will get after prewhere and filter executions.
            NamesAndTypesList required_columns_after_prewhere;
            NameSet required_columns_after_prewhere_set;

            /// Collect required columns from prewhere expression actions.
            if (prewhere_info)
            {
                NameSet columns_to_remove(columns_to_remove_after_prewhere.begin(), columns_to_remove_after_prewhere.end());
                Block prewhere_actions_result = prewhere_info->prewhere_actions->getSampleBlock();

                /// Populate required columns with the columns, added by PREWHERE actions and not removed afterwards.
                /// XXX: looks hacky that we already know which columns after PREWHERE we won't need for sure.
                for (const auto & column : prewhere_actions_result)
                {
                    if (prewhere_info->remove_prewhere_column && column.name == prewhere_info->prewhere_column_name)
                        continue;

                    if (columns_to_remove.count(column.name))
                        continue;

                    required_columns_all_expr->children.emplace_back(std::make_shared<ASTIdentifier>(column.name));
                    required_columns_after_prewhere.emplace_back(column.name, column.type);
                }

                required_columns_after_prewhere_set
                    = ext::map<NameSet>(required_columns_after_prewhere, [](const auto & it) { return it.name; });
            }

            auto syntax_result = SyntaxAnalyzer(*context).analyze(required_columns_all_expr, required_columns_after_prewhere, storage);
            alias_actions = ExpressionAnalyzer(required_columns_all_expr, syntax_result, *context).getActions(true);

            /// The set of required columns could be added as a result of adding an action to calculate ALIAS.
            required_columns = alias_actions->getRequiredColumns();

            /// Do not remove prewhere filter if it is a column which is used as alias.
            if (prewhere_info && prewhere_info->remove_prewhere_column)
                if (required_columns.end()
                    != std::find(required_columns.begin(), required_columns.end(), prewhere_info->prewhere_column_name))
                    prewhere_info->remove_prewhere_column = false;

            /// Remove columns which will be added by prewhere.
            required_columns.erase(std::remove_if(required_columns.begin(), required_columns.end(), [&](const String & name)
            {
                return !!required_columns_after_prewhere_set.count(name);
            }), required_columns.end());

            if (prewhere_info)
            {
                /// Don't remove columns which are needed to be aliased.
                auto new_actions = std::make_shared<ExpressionActions>(prewhere_info->prewhere_actions->getRequiredColumnsWithTypes(), *context);
                for (const auto & action : prewhere_info->prewhere_actions->getActions())
                {
                    if (action.type != ExpressionAction::REMOVE_COLUMN
                        || required_columns.end() == std::find(required_columns.begin(), required_columns.end(), action.source_name))
                        new_actions->add(action);
                }
                prewhere_info->prewhere_actions = std::move(new_actions);

                auto analyzed_result
                    = SyntaxAnalyzer(*context).analyze(required_columns_from_prewhere_expr, storage->getColumns().getAllPhysical());
                prewhere_info->alias_actions
                    = ExpressionAnalyzer(required_columns_from_prewhere_expr, analyzed_result, *context).getActions(true, false);

                /// Add (physical?) columns required by alias actions.
                auto required_columns_from_alias = prewhere_info->alias_actions->getRequiredColumns();
                Block prewhere_actions_result = prewhere_info->prewhere_actions->getSampleBlock();
                for (auto & column : required_columns_from_alias)
                    if (!prewhere_actions_result.has(column))
                        if (required_columns.end() == std::find(required_columns.begin(), required_columns.end(), column))
                            required_columns.push_back(column);

                /// Add physical columns required by prewhere actions.
                for (const auto & column : required_columns_from_prewhere)
                    if (required_aliases_from_prewhere.count(column) == 0)
                        if (required_columns.end() == std::find(required_columns.begin(), required_columns.end(), column))
                            required_columns.push_back(column);
            }
        }
    }

    /// Limitation on the number of columns to read.
    /// It's not applied in 'only_analyze' mode, because the query could be analyzed without removal of unnecessary columns.
    if (!options.only_analyze && settings.max_columns_to_read && required_columns.size() > settings.max_columns_to_read)
        throw Exception("Limit for number of columns to read exceeded. "
            "Requested: " + toString(required_columns.size())
            + ", maximum: " + settings.max_columns_to_read.toString(),
            ErrorCodes::TOO_MANY_COLUMNS);

    /** With distributed query processing, almost no computations are done in the threads,
     *  but wait and receive data from remote servers.
     *  If we have 20 remote servers, and max_threads = 8, then it would not be very good
     *  connect and ask only 8 servers at a time.
     *  To simultaneously query more remote servers,
     *  instead of max_threads, max_distributed_connections is used.
     */
    bool is_remote = false;
    if (storage && storage->isRemote())
    {
        is_remote = true;
        max_streams = settings.max_distributed_connections;
        pipeline.setMaxThreads(max_streams);
    }

    UInt64 max_block_size = settings.max_block_size;

    auto [limit_length, limit_offset] = getLimitLengthAndOffset(query, *context);

    /** Optimization - if not specified DISTINCT, WHERE, GROUP, HAVING, ORDER, LIMIT BY, WITH TIES but LIMIT is specified, and limit + offset < max_block_size,
     *  then as the block size we will use limit + offset (not to read more from the table than requested),
     *  and also set the number of threads to 1.
     */
    if (!query.distinct
        && !query.limit_with_ties
        && !query.prewhere()
        && !query.where()
        && !query.groupBy()
        && !query.having()
        && !query.orderBy()
        && !query.limitBy()
        && query.limitLength()
        && !query_analyzer->hasAggregation()
        && limit_length + limit_offset < max_block_size)
    {
        max_block_size = std::max(UInt64(1), limit_length + limit_offset);
        max_streams = 1;
        pipeline.setMaxThreads(max_streams);
    }

    if (!max_block_size)
        throw Exception("Setting 'max_block_size' cannot be zero", ErrorCodes::PARAMETER_OUT_OF_BOUND);

    /// Initialize the initial data streams to which the query transforms are superimposed. Table or subquery or prepared input?
    if (pipeline.initialized())
    {
        /// Prepared input.
    }
    else if (interpreter_subquery)
    {
        /// Subquery.
        /// If we need less number of columns that subquery have - update the interpreter.
        if (required_columns.size() < source_header.columns())
        {
            ASTPtr subquery = extractTableExpression(query, 0);
            if (!subquery)
                throw Exception("Subquery expected", ErrorCodes::LOGICAL_ERROR);

            interpreter_subquery = std::make_unique<InterpreterSelectWithUnionQuery>(
                subquery, getSubqueryContext(*context),
                options.copy().subquery().noModify(), required_columns);

            if (query_analyzer->hasAggregation())
                interpreter_subquery->ignoreWithTotals();
        }

        if constexpr (pipeline_with_processors)
            /// Just use pipeline from subquery.
            pipeline = interpreter_subquery->executeWithProcessors();
        else
            pipeline.streams = interpreter_subquery->executeWithMultipleStreams(save_context_and_storage);
    }
    else if (storage)
    {
        /// Table.

        if (max_streams == 0)
            throw Exception("Logical error: zero number of streams requested", ErrorCodes::LOGICAL_ERROR);

        /// If necessary, we request more sources than the number of threads - to distribute the work evenly over the threads.
        if (max_streams > 1 && !is_remote)
            max_streams *= settings.max_streams_to_max_threads_ratio;

        query_info.query = query_ptr;
        query_info.syntax_analyzer_result = syntax_analyzer_result;
        query_info.sets = query_analyzer->getPreparedSets();
        query_info.prewhere_info = prewhere_info;

        /// Create optimizer with prepared actions.
        /// Maybe we will need to calc input_sorting_info later, e.g. while reading from StorageMerge.
        if (analysis_result.optimize_read_in_order)
        {
            query_info.order_by_optimizer = std::make_shared<ReadInOrderOptimizer>(
                analysis_result.order_by_elements_actions,
                getSortDescription(query, *context),
                query_info.syntax_analyzer_result);

            query_info.input_sorting_info = query_info.order_by_optimizer->getInputOrder(storage);
        }


        BlockInputStreams streams;
        Pipes pipes;

        if (pipeline_with_processors)
            pipes = storage->read(required_columns, query_info, *context, processing_stage, max_block_size, max_streams);
        else
            streams = storage->readStreams(required_columns, query_info, *context, processing_stage, max_block_size, max_streams);

        if (streams.empty() && !pipeline_with_processors)
        {
            streams = {std::make_shared<NullBlockInputStream>(storage->getSampleBlockForColumns(required_columns))};

            if (query_info.prewhere_info)
            {
                if (query_info.prewhere_info->alias_actions)
                {
                    streams.back() = std::make_shared<ExpressionBlockInputStream>(
                        streams.back(),
                        query_info.prewhere_info->alias_actions);
                }

                streams.back() = std::make_shared<FilterBlockInputStream>(
                    streams.back(),
                    prewhere_info->prewhere_actions,
                    prewhere_info->prewhere_column_name,
                    prewhere_info->remove_prewhere_column);

                // To remove additional columns
                // In some cases, we did not read any marks so that the pipeline.streams is empty
                // Thus, some columns in prewhere are not removed as expected
                // This leads to mismatched header in distributed table
                if (query_info.prewhere_info->remove_columns_actions)
                {
                    streams.back() = std::make_shared<ExpressionBlockInputStream>(streams.back(), query_info.prewhere_info->remove_columns_actions);
                }
            }
        }

        /// Copy-paste from prev if.
        /// Code is temporarily copy-pasted while moving to new pipeline.
        if (pipes.empty() && pipeline_with_processors)
        {
            Pipe pipe(std::make_shared<NullSource>(storage->getSampleBlockForColumns(required_columns)));

            if (query_info.prewhere_info)
            {
                if (query_info.prewhere_info->alias_actions)
                    pipe.addSimpleTransform(std::make_shared<ExpressionTransform>(
                        pipe.getHeader(), query_info.prewhere_info->alias_actions));

                pipe.addSimpleTransform(std::make_shared<FilterTransform>(
                        pipe.getHeader(),
                        prewhere_info->prewhere_actions,
                        prewhere_info->prewhere_column_name,
                        prewhere_info->remove_prewhere_column));

                if (query_info.prewhere_info->remove_columns_actions)
                    pipe.addSimpleTransform(std::make_shared<ExpressionTransform>(pipe.getHeader(), query_info.prewhere_info->remove_columns_actions));
            }

            pipes.emplace_back(std::move(pipe));
        }

        for (auto & stream : streams)
            stream->addTableLock(table_lock);

        if constexpr (pipeline_with_processors)
        {
            /// Table lock is stored inside pipeline here.
            pipeline.addTableLock(table_lock);
        }

        /// Set the limits and quota for reading data, the speed and time of the query.
        {
            IBlockInputStream::LocalLimits limits;
            limits.mode = IBlockInputStream::LIMITS_TOTAL;
            limits.size_limits = SizeLimits(settings.max_rows_to_read, settings.max_bytes_to_read, settings.read_overflow_mode);
            limits.speed_limits.max_execution_time = settings.max_execution_time;
            limits.timeout_overflow_mode = settings.timeout_overflow_mode;

            /** Quota and minimal speed restrictions are checked on the initiating server of the request, and not on remote servers,
              *  because the initiating server has a summary of the execution of the request on all servers.
              *
              * But limits on data size to read and maximum execution time are reasonable to check both on initiator and
              *  additionally on each remote server, because these limits are checked per block of data processed,
              *  and remote servers may process way more blocks of data than are received by initiator.
              */
            if (options.to_stage == QueryProcessingStage::Complete)
            {
                limits.speed_limits.min_execution_rps = settings.min_execution_speed;
                limits.speed_limits.max_execution_rps = settings.max_execution_speed;
                limits.speed_limits.min_execution_bps = settings.min_execution_speed_bytes;
                limits.speed_limits.max_execution_bps = settings.max_execution_speed_bytes;
                limits.speed_limits.timeout_before_checking_execution_speed = settings.timeout_before_checking_execution_speed;
            }

            auto quota = context->getQuota();

            for (auto & stream : streams)
            {
                if (!options.ignore_limits)
                    stream->setLimits(limits);

                if (!options.ignore_quota && (options.to_stage == QueryProcessingStage::Complete))
                    stream->setQuota(quota);
            }

            /// Copy-paste
            for (auto & pipe : pipes)
            {
                if (!options.ignore_limits)
                    pipe.setLimits(limits);

                if (!options.ignore_quota && (options.to_stage == QueryProcessingStage::Complete))
                    pipe.setQuota(quota);
            }
        }

        if constexpr (pipeline_with_processors)
        {
            if (streams.size() == 1 || pipes.size() == 1)
                pipeline.setMaxThreads(1);

            /// Unify streams. They must have same headers.
            if (streams.size() > 1)
            {
                /// Unify streams in case they have different headers.
                auto first_header = streams.at(0)->getHeader();

                if (first_header.columns() > 1 && first_header.has("_dummy"))
                    first_header.erase("_dummy");

                for (auto & stream : streams)
                {
                    auto header = stream->getHeader();
                    auto mode = ConvertingBlockInputStream::MatchColumnsMode::Name;
                    if (!blocksHaveEqualStructure(first_header, header))
                        stream = std::make_shared<ConvertingBlockInputStream>(stream, first_header, mode);
                }
            }

            for (auto & stream : streams)
            {
                bool force_add_agg_info = processing_stage == QueryProcessingStage::WithMergeableState;
