template <typename T>
int ColumnDecimal<T>::compareAt(size_t n, size_t m, const IColumn & rhs_, int) const
{
    auto & other = static_cast<const Self &>(rhs_);
    const T & a = data[n];
    const T & b = other.data[m];

    if (scale == other.scale)
        return a > b ? 1 : (a < b ? -1 : 0);
    return decimalLess<T>(b, a, other.scale, scale) ? 1 : (decimalLess<T>(a, b, scale, other.scale) ? -1 : 0);
}
