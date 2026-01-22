void ArrowColumnToCHColumn::arrowColumnsToCHChunk(Chunk & res, NameToColumnPtr & name_to_column_ptr)
{
    if (name_to_column_ptr.empty())
        throw Exception(ErrorCodes::INCORRECT_NUMBER_OF_COLUMNS, "Columns is empty");

    Columns columns_list;
    UInt64 num_rows = name_to_column_ptr.begin()->second->length();
    columns_list.reserve(header.rows());
    std::unordered_map<String, BlockPtr> nested_tables;
    for (size_t column_i = 0, columns = header.columns(); column_i < columns; ++column_i)
    {
        const ColumnWithTypeAndName & header_column = header.getByPosition(column_i);

        bool read_from_nested = false;
        String nested_table_name = Nested::extractTableName(header_column.name);
        if (!name_to_column_ptr.contains(header_column.name))
        {
            /// Check if it's a column from nested table.
            if (import_nested && name_to_column_ptr.contains(nested_table_name))
            {
                if (!nested_tables.contains(nested_table_name))
                {
                    std::shared_ptr<arrow::ChunkedArray> arrow_column = name_to_column_ptr[nested_table_name];
                    ColumnsWithTypeAndName cols = {readColumnFromArrowColumn(arrow_column, nested_table_name, format_name, false, dictionary_values, true)};
                    Block block(cols);
                    nested_tables[nested_table_name] = std::make_shared<Block>(Nested::flatten(block));
                }

                read_from_nested = nested_tables[nested_table_name]->has(header_column.name);
            }

            if (!read_from_nested)
            {
                if (!allow_missing_columns)
                    throw Exception{ErrorCodes::THERE_IS_NO_COLUMN, "Column '{}' is not presented in input data.", header_column.name};

                ColumnWithTypeAndName column;
                column.name = header_column.name;
                column.type = header_column.type;
                column.column = header_column.column->cloneResized(num_rows);
                columns_list.push_back(std::move(column.column));
                continue;
            }
        }

        std::shared_ptr<arrow::ChunkedArray> arrow_column = name_to_column_ptr[header_column.name];

        ColumnWithTypeAndName column;
        if (read_from_nested)
            column = nested_tables[nested_table_name]->getByName(header_column.name);
        else
            column = readColumnFromArrowColumn(arrow_column, header_column.name, format_name, false, dictionary_values, true);

        try
        {
            column.column = castColumn(column, header_column.type);
        }
        catch (Exception & e)
        {
            e.addMessage(fmt::format("while converting column {} from type {} to type {}",
                backQuote(header_column.name), column.type->getName(), header_column.type->getName()));
            throw;
        }

        column.type = header_column.type;
        columns_list.push_back(std::move(column.column));
    }

    res.setColumns(columns_list, num_rows);
}
