ASTPtr MutationsInterpreter::prepare(bool dry_run)
{
    if (is_prepared)
        throw Exception("MutationsInterpreter is already prepared. It is a bug.", ErrorCodes::LOGICAL_ERROR);

    if (commands.empty())
        throw Exception("Empty mutation commands list", ErrorCodes::LOGICAL_ERROR);

    const ColumnsDescription & columns_desc = metadata_snapshot->getColumns();
    const IndicesDescription & indices_desc = metadata_snapshot->getSecondaryIndices();
    const ProjectionsDescription & projections_desc = metadata_snapshot->getProjections();
    NamesAndTypesList all_columns = columns_desc.getAllPhysical();

    NameSet updated_columns;
    bool materialize_ttl_recalculate_only = materializeTTLRecalculateOnly(storage);

    for (const MutationCommand & command : commands)
    {
        if (command.type == MutationCommand::Type::UPDATE
            || command.type == MutationCommand::Type::DELETE)
            materialize_ttl_recalculate_only = false;

        for (const auto & kv : command.column_to_update_expression)
        {
            updated_columns.insert(kv.first);
        }
    }

    /// We need to know which columns affect which MATERIALIZED columns, data skipping indices
    /// and projections to recalculate them if dependencies are updated.
    std::unordered_map<String, Names> column_to_affected_materialized;
    if (!updated_columns.empty())
    {
        for (const auto & column : columns_desc)
        {
            if (column.default_desc.kind == ColumnDefaultKind::Materialized)
            {
                auto query = column.default_desc.expression->clone();
                auto syntax_result = TreeRewriter(context).analyze(query, all_columns);
                for (const String & dependency : syntax_result->requiredSourceColumns())
                {
                    if (updated_columns.contains(dependency))
                        column_to_affected_materialized[dependency].push_back(column.name);
                }
            }
        }

        validateUpdateColumns(storage, metadata_snapshot, updated_columns, column_to_affected_materialized);
    }

    dependencies = getAllColumnDependencies(metadata_snapshot, updated_columns);

    /// First, break a sequence of commands into stages.
    for (auto & command : commands)
    {
        // we can return deleted rows only if it's the only present command
        assert(command.type == MutationCommand::DELETE || !return_deleted_rows);

        if (command.type == MutationCommand::DELETE)
        {
            mutation_kind.set(MutationKind::MUTATE_OTHER);
            if (stages.empty() || !stages.back().column_to_updated.empty())
                stages.emplace_back(context);

            auto predicate  = getPartitionAndPredicateExpressionForMutationCommand(command);

            if (!return_deleted_rows)
                predicate = makeASTFunction("isZeroOrNull", predicate);

            stages.back().filters.push_back(predicate);
        }
        else if (command.type == MutationCommand::UPDATE)
        {
            mutation_kind.set(MutationKind::MUTATE_OTHER);
            if (stages.empty() || !stages.back().column_to_updated.empty())
                stages.emplace_back(context);
            if (stages.size() == 1) /// First stage only supports filtering and can't update columns.
                stages.emplace_back(context);

            NameSet affected_materialized;

            for (const auto & kv : command.column_to_update_expression)
            {
                const String & column = kv.first;

                auto materialized_it = column_to_affected_materialized.find(column);
                if (materialized_it != column_to_affected_materialized.end())
                {
                    for (const String & mat_column : materialized_it->second)
                        affected_materialized.emplace(mat_column);
                }

                /// When doing UPDATE column = expression WHERE condition
                /// we will replace column to the result of the following expression:
                ///
                /// CAST(if(condition, CAST(expression, type), column), type)
                ///
                /// Inner CAST is needed to make 'if' work when branches have no common type,
                /// example: type is UInt64, UPDATE x = -1 or UPDATE x = x - 1.
                ///
                /// Outer CAST is added just in case if we don't trust the returning type of 'if'.

                DataTypePtr type;
                if (auto physical_column = columns_desc.tryGetPhysical(column))
                    type = physical_column->type;
                else if (column == LightweightDeleteDescription::FILTER_COLUMN.name)
                    type = LightweightDeleteDescription::FILTER_COLUMN.type;
                else
                    throw Exception(ErrorCodes::LOGICAL_ERROR, "Unknown column {}", column);

                auto type_literal = std::make_shared<ASTLiteral>(type->getName());

                const auto & update_expr = kv.second;

                ASTPtr condition = getPartitionAndPredicateExpressionForMutationCommand(command);

                /// And new check validateNestedArraySizes for Nested subcolumns
                if (isArray(type) && !Nested::splitName(column).second.empty())
                {
                    std::shared_ptr<ASTFunction> function = nullptr;

                    auto nested_update_exprs = getExpressionsOfUpdatedNestedSubcolumns(column, all_columns, command.column_to_update_expression);
                    if (!nested_update_exprs)
                    {
                        function = makeASTFunction("validateNestedArraySizes",
                            condition,
                            update_expr->clone(),
                            std::make_shared<ASTIdentifier>(column));
                        condition = makeASTFunction("and", condition, function);
                    }
                    else if (nested_update_exprs->size() > 1)
                    {
                        function = std::make_shared<ASTFunction>();
                        function->name = "validateNestedArraySizes";
                        function->arguments = std::make_shared<ASTExpressionList>();
                        function->children.push_back(function->arguments);
                        function->arguments->children.push_back(condition);
                        for (const auto & it : *nested_update_exprs)
                            function->arguments->children.push_back(it->clone());
                        condition = makeASTFunction("and", condition, function);
                    }
                }

                auto updated_column = makeASTFunction("_CAST",
                    makeASTFunction("if",
                        condition,
                        makeASTFunction("_CAST",
                            update_expr->clone(),
                            type_literal),
                        std::make_shared<ASTIdentifier>(column)),
                    type_literal);

                stages.back().column_to_updated.emplace(column, updated_column);
            }

            if (!affected_materialized.empty())
            {
                stages.emplace_back(context);
                for (const auto & column : columns_desc)
                {
                    if (column.default_desc.kind == ColumnDefaultKind::Materialized)
                    {
                        stages.back().column_to_updated.emplace(
                            column.name,
                            column.default_desc.expression->clone());
                    }
                }
            }
        }
        else if (command.type == MutationCommand::MATERIALIZE_COLUMN)
        {
            mutation_kind.set(MutationKind::MUTATE_OTHER);
            if (stages.empty() || !stages.back().column_to_updated.empty())
                stages.emplace_back(context);
            if (stages.size() == 1) /// First stage only supports filtering and can't update columns.
                stages.emplace_back(context);

            const auto & column = columns_desc.get(command.column_name);

            if (!column.default_desc.expression)
                throw Exception(
                    ErrorCodes::BAD_ARGUMENTS,
                    "Cannot materialize column `{}` because it doesn't have default expression", column.name);

            auto materialized_column = makeASTFunction(
                "_CAST", column.default_desc.expression->clone(), std::make_shared<ASTLiteral>(column.type->getName()));

            stages.back().column_to_updated.emplace(column.name, materialized_column);
        }
        else if (command.type == MutationCommand::MATERIALIZE_INDEX)
        {
            mutation_kind.set(MutationKind::MUTATE_INDEX_PROJECTION);
            auto it = std::find_if(
                    std::cbegin(indices_desc), std::end(indices_desc),
                    [&](const IndexDescription & index)
                    {
                        return index.name == command.index_name;
                    });
            if (it == std::cend(indices_desc))
                throw Exception("Unknown index: " + command.index_name, ErrorCodes::BAD_ARGUMENTS);

            auto query = (*it).expression_list_ast->clone();
            auto syntax_result = TreeRewriter(context).analyze(query, all_columns);
            const auto required_columns = syntax_result->requiredSourceColumns();
            for (const auto & column : required_columns)
                dependencies.emplace(column, ColumnDependency::SKIP_INDEX);
            materialized_indices.emplace(command.index_name);
        }
        else if (command.type == MutationCommand::MATERIALIZE_PROJECTION)
        {
            mutation_kind.set(MutationKind::MUTATE_INDEX_PROJECTION);
            const auto & projection = projections_desc.get(command.projection_name);
            for (const auto & column : projection.required_columns)
                dependencies.emplace(column, ColumnDependency::PROJECTION);
            materialized_projections.emplace(command.projection_name);
        }
        else if (command.type == MutationCommand::DROP_INDEX)
        {
            mutation_kind.set(MutationKind::MUTATE_INDEX_PROJECTION);
            materialized_indices.erase(command.index_name);
        }
        else if (command.type == MutationCommand::DROP_PROJECTION)
        {
            mutation_kind.set(MutationKind::MUTATE_INDEX_PROJECTION);
            materialized_projections.erase(command.projection_name);
        }
        else if (command.type == MutationCommand::MATERIALIZE_TTL)
        {
            mutation_kind.set(MutationKind::MUTATE_OTHER);
            if (materialize_ttl_recalculate_only)
            {
                // just recalculate ttl_infos without remove expired data
                auto all_columns_vec = all_columns.getNames();
                auto new_dependencies = metadata_snapshot->getColumnDependencies(NameSet(all_columns_vec.begin(), all_columns_vec.end()), false);
                for (const auto & dependency : new_dependencies)
                {
                    if (dependency.kind == ColumnDependency::TTL_EXPRESSION)
                        dependencies.insert(dependency);
                }
            }
            else if (metadata_snapshot->hasRowsTTL()
                || metadata_snapshot->hasAnyRowsWhereTTL()
                || metadata_snapshot->hasAnyGroupByTTL())
            {
                for (const auto & column : all_columns)
                    dependencies.emplace(column.name, ColumnDependency::TTL_TARGET);
            }
            else
            {
                NameSet new_updated_columns;
                auto column_ttls = metadata_snapshot->getColumns().getColumnTTLs();
                for (const auto & elem : column_ttls)
                {
                    dependencies.emplace(elem.first, ColumnDependency::TTL_TARGET);
                    new_updated_columns.insert(elem.first);
                }

                auto all_columns_vec = all_columns.getNames();
                auto all_dependencies = getAllColumnDependencies(metadata_snapshot, NameSet(all_columns_vec.begin(), all_columns_vec.end()));

                for (const auto & dependency : all_dependencies)
                {
                    if (dependency.kind == ColumnDependency::TTL_EXPRESSION)
                        dependencies.insert(dependency);
                }

                /// Recalc only skip indices and projections of columns which could be updated by TTL.
                auto new_dependencies = metadata_snapshot->getColumnDependencies(new_updated_columns, true);
                for (const auto & dependency : new_dependencies)
                {
                    if (dependency.kind == ColumnDependency::SKIP_INDEX || dependency.kind == ColumnDependency::PROJECTION)
                        dependencies.insert(dependency);
                }
            }

            if (dependencies.empty())
            {
                /// Very rare case. It can happen if we have only one MOVE TTL with constant expression.
                /// But we still have to read at least one column.
                dependencies.emplace(all_columns.front().name, ColumnDependency::TTL_EXPRESSION);
            }
        }
        else if (command.type == MutationCommand::READ_COLUMN)
        {
            mutation_kind.set(MutationKind::MUTATE_OTHER);
            if (stages.empty() || !stages.back().column_to_updated.empty())
                stages.emplace_back(context);
            if (stages.size() == 1) /// First stage only supports filtering and can't update columns.
                stages.emplace_back(context);

            stages.back().column_to_updated.emplace(command.column_name, std::make_shared<ASTIdentifier>(command.column_name));
        }
        else
            throw Exception("Unknown mutation command type: " + DB::toString<int>(command.type), ErrorCodes::UNKNOWN_MUTATION_COMMAND);
    }

    /// We care about affected indices and projections because we also need to rewrite them
    /// when one of index columns updated or filtered with delete.
    /// The same about columns, that are needed for calculation of TTL expressions.
    if (!dependencies.empty())
    {
        NameSet changed_columns;
        NameSet unchanged_columns;
        for (const auto & dependency : dependencies)
        {
            if (dependency.isReadOnly())
                unchanged_columns.insert(dependency.column_name);
            else
                changed_columns.insert(dependency.column_name);
        }

        if (!changed_columns.empty())
        {
            if (stages.empty() || !stages.back().column_to_updated.empty())
                stages.emplace_back(context);
            if (stages.size() == 1) /// First stage only supports filtering and can't update columns.
                stages.emplace_back(context);

            for (const auto & column : changed_columns)
                stages.back().column_to_updated.emplace(
                    column, std::make_shared<ASTIdentifier>(column));
        }

        if (!unchanged_columns.empty())
        {
            if (!stages.empty())
            {
                std::vector<Stage> stages_copy;
                /// Copy all filled stages except index calculation stage.
                for (const auto & stage : stages)
                {
                    stages_copy.emplace_back(context);
                    stages_copy.back().column_to_updated = stage.column_to_updated;
                    stages_copy.back().output_columns = stage.output_columns;
                    stages_copy.back().filters = stage.filters;
                }

                const ASTPtr select_query = prepareInterpreterSelectQuery(stages_copy, /* dry_run = */ true);
                InterpreterSelectQuery interpreter{
                    select_query, context, storage, metadata_snapshot,
                    SelectQueryOptions().analyze(/* dry_run = */ false).ignoreLimits().ignoreProjections()};

                auto first_stage_header = interpreter.getSampleBlock();
                QueryPlan plan;
                auto source = std::make_shared<NullSource>(first_stage_header);
                plan.addStep(std::make_unique<ReadFromPreparedSource>(Pipe(std::move(source))));
                auto pipeline = addStreamsForLaterStages(stages_copy, plan);
                updated_header = std::make_unique<Block>(pipeline.getHeader());
            }

            /// Special step to recalculate affected indices, projections and TTL expressions.
            stages.emplace_back(context);
            for (const auto & column : unchanged_columns)
                stages.back().column_to_updated.emplace(
                    column, std::make_shared<ASTIdentifier>(column));
        }
    }

    is_prepared = true;

    return prepareInterpreterSelectQuery(stages, dry_run);
}
