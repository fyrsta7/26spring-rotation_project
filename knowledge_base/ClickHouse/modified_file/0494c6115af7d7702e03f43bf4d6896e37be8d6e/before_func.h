    template <typename LeftType, typename RightType>
    static ColumnPtr executeTyped(const ColumnConst * left_arg, const IColumn * right_arg)
    {
        if (const auto right_arg_typed = checkAndGetColumn<ColumnVectorOrDecimal<RightType>>(right_arg))
        {
            auto dst = ColumnVector<Float64>::create();
            auto & dst_data = dst->getData();
            const auto & right_src_data = right_arg_typed->getData();
            const auto src_size = right_src_data.size();
            dst_data.resize(src_size);

            if constexpr (is_decimal<LeftType>)
            {
                Float64 left_src_data[Impl::rows_per_iteration];
                const auto left_data_column = left_arg->getDataColumnPtr();
                const auto left_scale = checkAndGetColumn<ColumnDecimal<LeftType>>(*left_data_column).getScale();
                std::fill(std::begin(left_src_data), std::end(left_src_data), DecimalUtils::convertTo<Float64>(left_arg->template getValue<LeftType>(), left_scale));

                if constexpr (is_decimal<RightType>)
                {
                    const auto right_scale = right_arg_typed->getScale();
                    for (size_t i = 0; i < src_size; ++i)
                        dst_data[i] = DecimalUtils::convertTo<Float64>(right_src_data[i], right_scale);

                    executeInIterations(left_src_data, std::size(left_src_data), dst_data.data(), src_size, dst_data.data());
                }
                else
                {
                    executeInIterations(left_src_data, std::size(left_src_data), right_src_data.data(), src_size, dst_data.data());
                }
            }
            else
            {
                LeftType left_src_data[Impl::rows_per_iteration];
                std::fill(std::begin(left_src_data), std::end(left_src_data), left_arg->template getValue<LeftType>());

                if constexpr (is_decimal<RightType>)
                {
                    const auto right_scale = right_arg_typed->getScale();
                    for (size_t i = 0; i < src_size; ++i)
                        dst_data[i] = DecimalUtils::convertTo<Float64>(right_src_data[i], right_scale);

                    executeInIterations(left_src_data, std::size(left_src_data), dst_data.data(), src_size, dst_data.data());
                }
                else
                {
                    executeInIterations(left_src_data, std::size(left_src_data), right_src_data.data(), src_size, dst_data.data());
                }
            }

            return dst;
        }

        return nullptr;
    }
