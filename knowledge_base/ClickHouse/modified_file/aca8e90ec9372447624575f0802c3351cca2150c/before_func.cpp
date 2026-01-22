                "Must be UInt8.", arguments[0]->getName());

        return getLeastSupertype(DataTypes{arguments[1], arguments[2]});
    }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & args, const DataTypePtr & result_type, size_t input_rows_count) const override
    {
        ColumnsWithTypeAndName arguments = args;
        executeShortCircuitArguments(arguments);
        ColumnPtr res;
        if (   (res = executeForConstAndNullableCondition(arguments, result_type, input_rows_count))
            || (res = executeForNullThenElse(arguments, result_type, input_rows_count))
            || (res = executeForNullableThenElse(arguments, result_type, input_rows_count)))
            return res;

        const ColumnWithTypeAndName & arg_cond = arguments[0];
        const ColumnWithTypeAndName & arg_then = arguments[1];
        const ColumnWithTypeAndName & arg_else = arguments[2];

        /// A case for identical then and else (pointers are the same).
        if (arg_then.column.get() == arg_else.column.get())
        {
            /// Just point result to them.
            return arg_then.column;
        }

        const ColumnUInt8 * cond_col = typeid_cast<const ColumnUInt8 *>(arg_cond.column.get());
        const ColumnConst * cond_const_col = checkAndGetColumnConst<ColumnVector<UInt8>>(arg_cond.column.get());
        ColumnPtr materialized_cond_col;

        if (cond_const_col)
        {
            if (arg_then.type->equals(*arg_else.type))
            {
                return cond_const_col->getValue<UInt8>()
                    ? arg_then.column
                    : arg_else.column;
            }
            else
            {
                /// TODO why materialize condition
                materialized_cond_col = cond_const_col->convertToFullColumn();
                cond_col = typeid_cast<const ColumnUInt8 *>(&*materialized_cond_col);
            }
        }

        if (!cond_col)
            throw Exception(ErrorCodes::ILLEGAL_COLUMN, "Illegal column {} of first argument of function {}. "
                "Must be ColumnUInt8 or ColumnConstUInt8.", arg_cond.column->getName(), getName());

        auto call = [&](const auto & types) -> bool
        {
            using Types = std::decay_t<decltype(types)>;
            using T0 = typename Types::LeftType;
            using T1 = typename Types::RightType;

            res = executeTyped<T0, T1>(cond_col, arguments, result_type, input_rows_count);
            return res != nullptr;
        };

        DataTypePtr left_type = arg_then.type;
        DataTypePtr right_type = arg_else.type;

        if (const auto * left_array = checkAndGetDataType<DataTypeArray>(arg_then.type.get()))
            left_type = left_array->getNestedType();

        if (const auto * right_array = checkAndGetDataType<DataTypeArray>(arg_else.type.get()))
            right_type = right_array->getNestedType();

        /// Special case when one column is Integer and another is UInt64 that can be actually Int64.
        /// The result type for this case is Int64 and we need to change UInt64 type to Int64
        /// so the NumberTraits::ResultOfIf will return Int64 instead if Int128.
        if (isNativeInteger(left_type) && isUInt64ThatCanBeInt64(right_type))
            right_type = std::make_shared<DataTypeInt64>();
        else if (isNativeInteger(right_type) && isUInt64ThatCanBeInt64(left_type))
            left_type = std::make_shared<DataTypeInt64>();

        TypeIndex left_id = left_type->getTypeId();
        TypeIndex right_id = right_type->getTypeId();

        /// TODO optimize for map type
        /// TODO optimize for nullable type
        if (!(callOnBasicTypes<true, true, true, false>(left_id, right_id, call)
            || (res = executeTyped<UUID, UUID>(cond_col, arguments, result_type, input_rows_count))
            || (res = executeString(cond_col, arguments, result_type))
            || (res = executeGenericArray(cond_col, arguments, result_type))
            || (res = executeTuple(arguments, result_type, input_rows_count))))
        {
