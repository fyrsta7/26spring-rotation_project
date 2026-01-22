            bitmap_function = AggregateFunctionFactory::instance().get(
                AggregateFunctionGroupBitmapData<UInt32>::name(), argument_types, params_row, properties);
        else if (which.isUInt64())
            bitmap_function = AggregateFunctionFactory::instance().get(
                AggregateFunctionGroupBitmapData<UInt64>::name(), argument_types, params_row, properties);
        else if (which.isInt8())
            bitmap_function = AggregateFunctionFactory::instance().get(
                AggregateFunctionGroupBitmapData<Int8>::name(), argument_types, params_row, properties);
        else if (which.isInt16())
            bitmap_function = AggregateFunctionFactory::instance().get(
                AggregateFunctionGroupBitmapData<Int16>::name(), argument_types, params_row, properties);
        else if (which.isInt32())
            bitmap_function = AggregateFunctionFactory::instance().get(
                AggregateFunctionGroupBitmapData<Int32>::name(), argument_types, params_row, properties);
        else if (which.isInt64())
            bitmap_function = AggregateFunctionFactory::instance().get(
                AggregateFunctionGroupBitmapData<Int64>::name(), argument_types, params_row, properties);
        else
            throw Exception(
                "Unexpected type " + array_type->getName() + " of argument of function " + getName(), ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT);

        return std::make_shared<DataTypeAggregateFunction>(bitmap_function, argument_types, params_row);
    }

    bool useDefaultImplementationForConstants() const override { return true; }

    ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr &, size_t /* input_rows_count */) const override
    {
        const IDataType * from_type = arguments[0].type.get();
        const auto * array_type = typeid_cast<const DataTypeArray *>(from_type);
        const auto & nested_type = array_type->getNestedType();
