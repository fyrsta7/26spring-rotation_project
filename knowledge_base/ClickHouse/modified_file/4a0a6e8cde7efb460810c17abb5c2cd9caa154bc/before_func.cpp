bool TranslateQualifiedNamesMatcher::Data::matchColumnName(const String & name, const String & column_name, DataTypePtr column_type)
{
    if (name.size() < column_name.size())
        return false;

    if (std::strncmp(name.data(), column_name.data(), column_name.size()) != 0)
        return false;

    if (name.size() == column_name.size())
        return true;

    /// In case the type is named tuple, check the name recursively.
    if (const DataTypeTuple * type_tuple = typeid_cast<const DataTypeTuple *>(column_type.get()))
    {
        if (type_tuple->haveExplicitNames() && name.at(column_name.size()) == '.')
        {
            const Strings & names = type_tuple->getElementNames();
            const DataTypes & element_types = type_tuple->getElements();
            for (size_t i = 0; i < names.size(); ++i)
            {
                if (matchColumnName(name.substr(column_name.size() + 1, name.size() - column_name.size()), names[i], element_types[i]))
                {
                    return true;
                }
            }
        }
    }

    return false;
}
