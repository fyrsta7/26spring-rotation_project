void IPolygonDictionary::blockToAttributes(const DB::Block &block)
{
    const auto rows = block.rows();
    element_count += rows;
    for (size_t i = 0; i < attributes.size(); ++i)
    {
        const auto & column = block.safeGetByPosition(i + 1);
        if (attributes[i])
        {
            MutableColumnPtr mutated = std::move(*attributes[i]).mutate();
            mutated->insertRangeFrom(*column.column, 0, column.column->size());
            attributes[i] = std::move(mutated);
        }
        else
            attributes[i] = column.column;
    }
    polygons.reserve(polygons.size() + rows);

    const auto & key = block.safeGetByPosition(0).column;

    for (const auto row : ext::range(0, rows))
    {
        const auto & field = (*key)[row];
        // TODO: Get data more efficiently using
        polygons.push_back(fieldToMultiPolygon(field));
    }
}
