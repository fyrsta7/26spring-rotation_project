    }

    ColumnPtr executeIntervalTupleOfIntervalsPlusMinus(const ColumnsWithTypeAndName & arguments, const DataTypePtr & result_type,
                                               size_t input_rows_count, const FunctionOverloadResolverPtr & function_builder) const
    {
        auto function = function_builder->build(arguments);

        return function->execute(arguments, result_type, input_rows_count);
    }

    ColumnPtr executeArrayImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr & result_type, size_t input_rows_count) const
    {
        const auto * return_type_array = checkAndGetDataType<DataTypeArray>(result_type.get());

        if (!return_type_array)
            throw Exception(ErrorCodes::LOGICAL_ERROR, "Return type for function {} must be array.", getName());

        auto num_args = arguments.size();
        DataTypes data_types;

        ColumnsWithTypeAndName new_arguments {num_args};
        DataTypePtr result_array_type;

        const auto * left_const = typeid_cast<const ColumnConst *>(arguments[0].column.get());
        const auto * right_const = typeid_cast<const ColumnConst *>(arguments[1].column.get());

        /// Unpacking arrays if both are constants.
        if (left_const && right_const)
        {
            new_arguments[0] = {left_const->getDataColumnPtr(), arguments[0].type, arguments[0].name};
            new_arguments[1] = {right_const->getDataColumnPtr(), arguments[1].type, arguments[1].name};
            auto col = executeImpl(new_arguments, result_type, 1);
            return ColumnConst::create(std::move(col), input_rows_count);
        }

        /// Unpacking arrays if at least one column is constant.
        if (left_const || right_const)
        {
            new_arguments[0] = {arguments[0].column->convertToFullColumnIfConst(), arguments[0].type, arguments[0].name};
            new_arguments[1] = {arguments[1].column->convertToFullColumnIfConst(), arguments[1].type, arguments[1].name};
            return executeImpl(new_arguments, result_type, input_rows_count);
        }

        const auto * left_array_col = typeid_cast<const ColumnArray *>(arguments[0].column.get());
        const auto * right_array_col = typeid_cast<const ColumnArray *>(arguments[1].column.get());
        if (!left_array_col->hasEqualOffsets(*right_array_col))
            throw Exception(ErrorCodes::SIZES_OF_ARRAYS_DONT_MATCH, "Two arguments for function {} must have equal sizes", getName());

        const auto & left_array_type = typeid_cast<const DataTypeArray *>(arguments[0].type.get())->getNestedType();
        new_arguments[0] = {left_array_col->getDataPtr(), left_array_type, arguments[0].name};

        const auto & right_array_type = typeid_cast<const DataTypeArray *>(arguments[1].type.get())->getNestedType();
        new_arguments[1] = {right_array_col->getDataPtr(), right_array_type, arguments[1].name};

