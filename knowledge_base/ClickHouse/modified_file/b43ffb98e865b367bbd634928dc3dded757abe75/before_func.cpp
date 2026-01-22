FilterDescription::FilterDescription(const IColumn & column_)
{
    if (column_.isSparse())
        data_holder = recursiveRemoveSparse(column_.getPtr());

    if (column_.lowCardinality())
        data_holder = column_.convertToFullColumnIfLowCardinality();

    const auto & column = data_holder ? *data_holder : column_;

    if (const ColumnUInt8 * concrete_column = typeid_cast<const ColumnUInt8 *>(&column))
    {
        data = &concrete_column->getData();
        return;
    }

    if (const auto * nullable_column = checkAndGetColumn<ColumnNullable>(column))
    {
        ColumnPtr nested_column = nullable_column->getNestedColumnPtr();
        MutableColumnPtr mutable_holder = IColumn::mutate(std::move(nested_column));

        ColumnUInt8 * concrete_column = typeid_cast<ColumnUInt8 *>(mutable_holder.get());
        if (!concrete_column)
            throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_COLUMN_FOR_FILTER,
                "Illegal type {} of column for filter. Must be UInt8 or Nullable(UInt8).", column.getName());

        const NullMap & null_map = nullable_column->getNullMapData();
        IColumn::Filter & res = concrete_column->getData();

        const auto size = res.size();
        assert(size == null_map.size());
        for (size_t i = 0; i < size; ++i)
            res[i] = res[i] && !null_map[i];

        data = &res;
        data_holder = std::move(mutable_holder);
        return;
    }

    throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_COLUMN_FOR_FILTER,
        "Illegal type {} of column for filter. Must be UInt8 or Nullable(UInt8) or Const variants of them.",
        column.getName());
}
