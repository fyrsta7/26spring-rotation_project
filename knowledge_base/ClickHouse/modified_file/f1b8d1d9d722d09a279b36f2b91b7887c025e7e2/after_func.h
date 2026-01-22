template <typename FromDataType, typename ToDataType, typename ReturnType = void>
requires (IsDataTypeDecimal<FromDataType> && IsDataTypeDecimal<ToDataType>)
inline ReturnType convertDecimalsImpl(const typename FromDataType::FieldType & value, UInt32 scale_from, UInt32 scale_to, typename ToDataType::FieldType & result)
{
    using FromFieldType = typename FromDataType::FieldType;
    using ToFieldType = typename ToDataType::FieldType;
    using MaxFieldType = std::conditional_t<(sizeof(FromFieldType) > sizeof(ToFieldType)), FromFieldType, ToFieldType>;
    using MaxNativeType = typename MaxFieldType::NativeType;

    static constexpr bool throw_exception = std::is_same_v<ReturnType, void>;

    MaxNativeType converted_value;
    if (scale_to > scale_from)
    {
        converted_value = DecimalUtils::scaleMultiplier<MaxNativeType>(scale_to - scale_from);
        if (common::mulOverflow(static_cast<MaxNativeType>(value.value), converted_value, converted_value))
        {
            if constexpr (throw_exception)
                throw Exception(ErrorCodes::DECIMAL_OVERFLOW, "{} convert overflow", std::string(ToDataType::family_name));
            else
                return ReturnType(false);
        }
    }
    else if (scale_to == scale_from)
    {
        converted_value = value.value;
    }
    else
    {
        converted_value = value.value / DecimalUtils::scaleMultiplier<MaxNativeType>(scale_from - scale_to);
    }

    if constexpr (sizeof(FromFieldType) > sizeof(ToFieldType))
    {
        if (converted_value < std::numeric_limits<typename ToFieldType::NativeType>::min() ||
            converted_value > std::numeric_limits<typename ToFieldType::NativeType>::max())
        {
            if constexpr (throw_exception)
                throw Exception(ErrorCodes::DECIMAL_OVERFLOW, "{} convert overflow", std::string(ToDataType::family_name));
            else
                return ReturnType(false);
        }
    }

    result = static_cast<typename ToFieldType::NativeType>(converted_value);

    return ReturnType(true);
}
