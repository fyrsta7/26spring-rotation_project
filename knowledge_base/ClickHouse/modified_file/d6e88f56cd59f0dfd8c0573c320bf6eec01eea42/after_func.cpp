void ColumnsDescription::addSubcolumns(const String & name_in_storage, const DataTypePtr & type_in_storage)
{
    IDataType::forEachSubcolumn([&](const auto &, const auto & subname, const auto & subdata)
    {
        auto subcolumn = NameAndTypePair(name_in_storage, subname, type_in_storage, subdata.type);

        if (has(subcolumn.name))
            throw Exception(ErrorCodes::ILLEGAL_COLUMN,
                "Cannot add subcolumn {}: column with this name already exists", subcolumn.name);

        subcolumns.get<0>().insert(std::move(subcolumn));
    }, {type_in_storage->getDefaultSerialization(), type_in_storage, nullptr, nullptr});
}
