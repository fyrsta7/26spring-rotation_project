        ColumnPtr executeImpl(const ColumnsWithTypeAndName & arguments, const DataTypePtr &, size_t) const override
        {
            if (arguments.size() != 2)
                throw Exception(ErrorCodes::NUMBER_OF_ARGUMENTS_DOESNT_MATCH, "Function {}'s arguments number must be 2.", name);
            const ColumnWithTypeAndName & arg1 = arguments[0];
            const ColumnWithTypeAndName & arg2 = arguments[1];
            const auto * time_zone_const_col = checkAndGetColumnConstData<ColumnString>(arg2.column.get());
            if (!time_zone_const_col)
                throw Exception(ErrorCodes::ILLEGAL_COLUMN, "Illegal column {} of 2nd argument of function {}. Excepted const(String).", arg2.column->getName(), name);
            String time_zone_val = time_zone_const_col->getDataAt(0).toString();
            if (WhichDataType(arg1.type).isDateTime())
            {
                const auto * date_time_col = checkAndGetColumn<ColumnDateTime>(arg1.column.get());
                size_t col_size = date_time_col->size();
                using ColVecTo = DataTypeDateTime::ColumnType;
                typename ColVecTo::MutablePtr result_column = ColVecTo::create(col_size);
                typename ColVecTo::Container & result_data = result_column->getData();
                for (size_t i = 0; i < col_size; ++i)
                {
                    UInt32 date_time_val = date_time_col->getElement(i);
                    LocalDateTime date_time(date_time_val, Name::to ? DateLUT::instance("UTC") : DateLUT::instance(time_zone_val));
                    time_t time_val = date_time.to_time_t(Name::from ? DateLUT::instance("UTC") : DateLUT::instance(time_zone_val));
                    result_data[i] = static_cast<UInt32>(time_val);
                }
                return result_column;
            }
            else if (WhichDataType(arg1.type).isDateTime64())
            {
                const auto * date_time_col = checkAndGetColumn<ColumnDateTime64>(arg1.column.get());
                size_t col_size = date_time_col->size();
                const DataTypeDateTime64 * date_time_type = static_cast<const DataTypeDateTime64 *>(arg1.type.get());
                UInt32 col_scale = date_time_type->getScale();
                Int64 scale_multiplier = DecimalUtils::scaleMultiplier<Int64>(col_scale);
                using ColDecimalTo = DataTypeDateTime64::ColumnType;
                typename ColDecimalTo::MutablePtr result_column = ColDecimalTo::create(col_size, col_scale);
                typename ColDecimalTo::Container & result_data = result_column->getData();
                for (size_t i = 0; i < col_size; ++i)
                {
                    DateTime64 date_time_val = date_time_col->getElement(i);
                    Int64 seconds = date_time_val.value / scale_multiplier;
                    Int64 micros = date_time_val.value % scale_multiplier;
                    LocalDateTime date_time(seconds, Name::to ? DateLUT::instance("UTC") : DateLUT::instance(time_zone_val));
                    time_t time_val = date_time.to_time_t(Name::from ? DateLUT::instance("UTC") : DateLUT::instance(time_zone_val));
                    DateTime64 date_time_64(time_val * scale_multiplier + micros);
                    result_data[i] = date_time_64;
                }
                return result_column;
            }
            else
                throw Exception(ErrorCodes::ILLEGAL_TYPE_OF_ARGUMENT, "Function {}'s 1st argument can only be datetime/datatime64. ", name);
        }
