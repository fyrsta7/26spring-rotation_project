void ColumnsDescription::addSubcolumns(const String & name_in_storage, const DataTypePtr & type_in_storage)
{
    for (const auto & subcolumn_name : type_in_storage->getSubcolumnNames())
    {
        auto subcolumn = NameAndTypePair(name_in_storage, subcolumn_name,
            type_in_storage, type_in_storage->getSubcolumnType(subcolumn_name));

        if (has(subcolumn.name))
            throw Exception(ErrorCodes::ILLEGAL_COLUMN,
                "Cannot add subcolumn {}: column with this name already exists", subcolumn.name);

        subcolumns.get<0>().insert(std::move(subcolumn));
    }
}
