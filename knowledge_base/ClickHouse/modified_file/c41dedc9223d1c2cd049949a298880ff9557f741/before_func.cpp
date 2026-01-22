bool ColumnLowCardinality::hasEqualValues() const
{
    return hasEqualValuesImpl<ColumnLowCardinality>();
}
