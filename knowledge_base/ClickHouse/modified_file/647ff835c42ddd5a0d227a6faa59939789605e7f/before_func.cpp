Block NumbersBlockInputStream::readImpl()
{
	Block res;
	
	ColumnWithNameAndType column_with_name_and_type;
	
	column_with_name_and_type.name = "number";
	column_with_name_and_type.type = new DataTypeUInt64();
	ColumnUInt64 * column = new ColumnUInt64(block_size);
	ColumnUInt64::Container_t & vec = column->getData();
	column_with_name_and_type.column = column;

	for (size_t i = 0; i < block_size; ++i)
		vec[i] = next++;

	res.insert(column_with_name_and_type);
	
	return res;
}
