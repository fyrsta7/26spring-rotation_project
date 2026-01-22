    template <typename AggregationKeyChecker>
    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, size_t input_rows_count, AggregationKeyChecker checker) const
    {
        auto grouping_set_column = checkAndGetColumn<ColumnUInt64>(arguments[0].column.get());

        auto result = std::make_shared<DataTypeUInt64>()->createColumn();
        for (size_t i = 0; i < input_rows_count; ++i)
        {
            UInt64 set_index = grouping_set_column->get64(i);

            UInt64 value = 0;
            for (auto index : arguments_indexes)
                value = (value << 1) + (checker(set_index, index) ? 1 : 0);

            result->insert(Field(value));
        }
        return result;
    }
