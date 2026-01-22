    template <typename AggregationKeyChecker>
    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, size_t input_rows_count, AggregationKeyChecker checker) const
    {
        const auto * grouping_set_column = checkAndGetColumn<ColumnUInt64>(arguments[0].column.get());

        auto result = ColumnUInt64::create();
        auto & result_data = result->getData();
        result_data.reserve(input_rows_count);
        for (size_t i = 0; i < input_rows_count; ++i)
        {
            UInt64 set_index = grouping_set_column->getElement(i);

            UInt64 value = 0;
            for (auto index : arguments_indexes)
                value = (value << 1) + (checker(set_index, index) ? 1 : 0);

            result_data.push_back(value);
        }
        return result;
    }
