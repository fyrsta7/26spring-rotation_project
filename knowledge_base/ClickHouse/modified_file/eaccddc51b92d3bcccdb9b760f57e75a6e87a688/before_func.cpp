
        /// Add values to the aggregate functions.
        for (AggregateFunctionInstruction * inst = aggregate_instructions; inst->that; ++inst)
            (*inst->func)(inst->that, value + inst->state_offset, inst->arguments, i, aggregates_pool);
    }
}


template <typename Method>
void NO_INLINE Aggregator::executeImplBatch(
    Method & method,
    typename Method::State & state,
    Arena * aggregates_pool,
    size_t rows,
    AggregateFunctionInstruction * aggregate_instructions) const
{
    if constexpr (std::is_same_v<Method, typename decltype(AggregatedDataVariants::key8)::element_type>)
    {
        for (AggregateFunctionInstruction * inst = aggregate_instructions; inst->that; ++inst)
        {
            inst->batch_that->addBatchLookupTable8(
                rows,
                reinterpret_cast<AggregateDataPtr *>(method.data.data()),
                inst->state_offset,
                [&](AggregateDataPtr & aggregate_data)
                {
                    aggregate_data = aggregates_pool->alignedAlloc(total_size_of_aggregate_states, align_aggregate_states);
                    createAggregateStates(aggregate_data);
                },
                state.getKeyData(),
                inst->batch_arguments,
                aggregates_pool);
        }
        return;
    }

    PODArray<AggregateDataPtr> places(rows);

    /// For all rows.
    for (size_t i = 0; i < rows; ++i)
    {
        AggregateDataPtr aggregate_data = nullptr;

        auto emplace_result = state.emplaceKey(method.data, i, *aggregates_pool);

        /// If a new key is inserted, initialize the states of the aggregate functions, and possibly something related to the key.
        if (emplace_result.isInserted())
        {
            /// exception-safety - if you can not allocate memory or create states, then destructors will not be called.
            emplace_result.setMapped(nullptr);

            aggregate_data = aggregates_pool->alignedAlloc(total_size_of_aggregate_states, align_aggregate_states);
            createAggregateStates(aggregate_data);

            emplace_result.setMapped(aggregate_data);
        }
        else
            aggregate_data = emplace_result.getMapped();

        places[i] = aggregate_data;
        assert(places[i] != nullptr);
    }

    /// Add values to the aggregate functions.
