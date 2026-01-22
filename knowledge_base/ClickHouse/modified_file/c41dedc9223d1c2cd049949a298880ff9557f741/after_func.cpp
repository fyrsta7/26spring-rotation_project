bool ColumnLowCardinality::hasEqualValues() const
{
    return getDictionary().size() <= 1;
}
