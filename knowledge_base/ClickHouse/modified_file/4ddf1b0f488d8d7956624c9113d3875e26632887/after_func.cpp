size_t tryPushDownFilter(QueryPlan::Node * parent_node, QueryPlan::Nodes & nodes)
{
    if (parent_node->children.size() != 1)
        return 0;

    QueryPlan::Node * child_node = parent_node->children.front();

    auto & parent = parent_node->step;
    auto & child = child_node->step;
    auto * filter = typeid_cast<FilterStep *>(parent.get());

    if (!filter)
        return 0;

    if (filter->getExpression()->hasStatefulFunctions())
        return 0;

    if (auto * aggregating = typeid_cast<AggregatingStep *>(child.get()))
    {
        const auto & params = aggregating->getParams();
        const auto & keys = params.keys;

        const bool filter_column_is_not_among_aggregation_keys
            = std::find(keys.begin(), keys.end(), filter->getFilterColumnName()) == keys.end();
        const bool can_remove_filter = filter_column_is_not_among_aggregation_keys
            && filterColumnIsNotAmongAggregatesArguments(params.aggregates, filter->getFilterColumnName());

        if (auto updated_steps = tryAddNewFilterStep(parent_node, nodes, keys, can_remove_filter))
            return updated_steps;
    }

    if (typeid_cast<CreatingSetsStep *>(child.get()))
    {
        /// CreatingSets does not change header.
        /// We can push down filter and update header.
        ///                       - Something
        /// Filter - CreatingSets - CreatingSet
        ///                       - CreatingSet
        auto input_streams = child->getInputStreams();
        input_streams.front() = filter->getOutputStream();
        child = std::make_unique<CreatingSetsStep>(input_streams);
        std::swap(parent, child);
        std::swap(parent_node->children, child_node->children);
        std::swap(parent_node->children.front(), child_node->children.front());
        ///              - Filter - Something
        /// CreatingSets - CreatingSet
        ///              - CreatingSet
        return 2;
    }

    if (auto * totals_having = typeid_cast<TotalsHavingStep *>(child.get()))
    {
        /// If totals step has HAVING expression, skip it for now.
        /// TODO:
        /// We can merge HAVING expression with current filer.
        /// Also, we can push down part of HAVING which depend only on aggregation keys.
        if (totals_having->getActions())
            return 0;

        Names keys;
        const auto & header = totals_having->getInputStreams().front().header;
        for (const auto & column : header)
            if (typeid_cast<const DataTypeAggregateFunction *>(column.type.get()) == nullptr)
                keys.push_back(column.name);

        /// NOTE: this optimization changes TOTALS value. Example:
        ///   `select * from (select y, sum(x) from (
        ///        select number as x, number % 4 as y from numbers(10)
        ///    ) group by y with totals) where y != 2`
        /// Optimization will replace totals row `y, sum(x)` from `(0, 45)` to `(0, 37)`.
        /// It is expected to ok, cause AST optimization `enable_optimize_predicate_expression = 1` also brakes it.
        if (auto updated_steps = tryAddNewFilterStep(parent_node, nodes, keys))
            return updated_steps;
    }

    if (auto * array_join = typeid_cast<ArrayJoinStep *>(child.get()))
    {
        const auto & array_join_actions = array_join->arrayJoin();
        const auto & keys = array_join_actions->columns;
        const auto & array_join_header = array_join->getInputStreams().front().header;

        Names allowed_inputs;
        for (const auto & column : array_join_header)
            if (!keys.contains(column.name))
                allowed_inputs.push_back(column.name);

        // for (const auto & name : allowed_inputs)
        //     std::cerr << name << std::endl;

        if (auto updated_steps = tryAddNewFilterStep(parent_node, nodes, allowed_inputs))
            return updated_steps;
    }

    if (auto updated_steps = simplePushDownOverStep<DistinctStep>(parent_node, nodes, child))
        return updated_steps;

    if (auto * join = typeid_cast<JoinStep *>(child.get()))
    {
        auto join_push_down = [&](JoinKind kind) -> size_t
        {
            const auto & table_join = join->getJoin()->getTableJoin();

            /// Only inner and left(/right) join are supported. Other types may generate default values for left table keys.
            /// So, if we push down a condition like `key != 0`, not all rows may be filtered.
            if (table_join.kind() != JoinKind::Inner && table_join.kind() != kind)
                return 0;

            bool is_left = kind == JoinKind::Left;
            const auto & input_header = is_left ? join->getInputStreams().front().header : join->getInputStreams().back().header;
            const auto & res_header = join->getOutputStream().header;
            Names allowed_keys;
            const auto & source_columns = input_header.getNames();
            for (const auto & name : source_columns)
            {
                /// Skip key if it is renamed.
                /// I don't know if it is possible. Just in case.
                if (!input_header.has(name) || !res_header.has(name))
                    continue;

                /// Skip if type is changed. Push down expression expect equal types.
                if (!input_header.getByName(name).type->equals(*res_header.getByName(name).type))
                    continue;

                allowed_keys.push_back(name);
            }

            /// For left JOIN, push down to the first child; for right - to the second one.
            const auto child_idx = is_left ? 0 : 1;
            ActionsDAGPtr split_filter = splitFilter(parent_node, allowed_keys, child_idx);
            if (!split_filter)
                return 0;
            /*
             * We should check the presence of a split filter column name in `source_columns` to avoid removing the required column.
             *
             * Example:
             * A filter expression is `a AND b = c`, but `b` and `c` belong to another side of the join and not in `allowed_keys`, so the final split filter is just `a`.
             * In this case `a` can be in `source_columns` but not `and(a, equals(b, c))`.
             *
             * New filter column is the first one.
             */
            const String & split_filter_column_name = split_filter->getOutputs().front()->result_name;
            bool can_remove_filter = source_columns.end() == std::find(source_columns.begin(), source_columns.end(), split_filter_column_name);
            const size_t updated_steps = tryAddNewFilterStep(parent_node, nodes, split_filter, can_remove_filter, child_idx);
            if (updated_steps > 0)
            {
                LOG_DEBUG(&Poco::Logger::get("QueryPlanOptimizations"), "Pushed down filter {} to the {} side of join", split_filter_column_name, kind);
            }
            return updated_steps;
        };

        if (size_t updated_steps = join_push_down(JoinKind::Left))
            return updated_steps;

        /// For full sorting merge join we push down both to the left and right tables, because left and right streams are not independent.
        if (join->allowPushDownToRight())
        {
            if (size_t updated_steps = join_push_down(JoinKind::Right))
                return updated_steps;
        }
    }

    /// TODO.
    /// We can filter earlier if expression does not depend on WITH FILL columns.
    /// But we cannot just push down condition, because other column may be filled with defaults.
    ///
    /// It is possible to filter columns before and after WITH FILL, but such change is not idempotent.
    /// So, appliying this to pair (Filter -> Filling) several times will create several similar filters.
    // if (auto * filling = typeid_cast<FillingStep *>(child.get()))
    // {
    // }

    /// Same reason for Cube
    // if (auto * cube = typeid_cast<CubeStep *>(child.get()))
    // {
    // }

    if (auto * sorting = typeid_cast<SortingStep *>(child.get()))
    {
        const auto & sort_description = sorting->getSortDescription();
        auto sort_description_it = std::find_if(sort_description.begin(), sort_description.end(), [&](auto & sort_column_description)
        {
            return sort_column_description.column_name == filter->getFilterColumnName();
        });
        bool can_remove_filter = sort_description_it == sort_description.end();

        Names allowed_inputs = child->getOutputStream().header.getNames();
        if (auto updated_steps = tryAddNewFilterStep(parent_node, nodes, allowed_inputs, can_remove_filter))
            return updated_steps;
    }

    if (auto updated_steps = simplePushDownOverStep<CreateSetAndFilterOnTheFlyStep>(parent_node, nodes, child))
        return updated_steps;

    if (auto * union_step = typeid_cast<UnionStep *>(child.get()))
    {
        /// Union does not change header.
        /// We can push down filter and update header.
        auto union_input_streams = child->getInputStreams();
        for (auto & input_stream : union_input_streams)
            input_stream.header = filter->getOutputStream().header;

        ///                - Something
        /// Filter - Union - Something
        ///                - Something

        child = std::make_unique<UnionStep>(union_input_streams, union_step->getMaxThreads());

        std::swap(parent, child);
        std::swap(parent_node->children, child_node->children);
        std::swap(parent_node->children.front(), child_node->children.front());

        ///       - Filter - Something
        /// Union - Something
        ///       - Something

        for (size_t i = 1; i < parent_node->children.size(); ++i)
        {
            auto & filter_node = nodes.emplace_back();
            filter_node.children.push_back(parent_node->children[i]);
            parent_node->children[i] = &filter_node;

            filter_node.step = std::make_unique<FilterStep>(
                filter_node.children.front()->step->getOutputStream(),
                filter->getExpression()->clone(),
                filter->getFilterColumnName(),
                filter->removesFilterColumn());
        }

        ///       - Filter - Something
