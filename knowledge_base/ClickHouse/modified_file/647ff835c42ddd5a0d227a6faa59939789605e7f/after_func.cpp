Block NumbersBlockInputStream::readImpl()
{
	Block res;
	
	ColumnWithNameAndType column_with_name_and_type;
	
	column_with_name_and_type.name = "number";
	column_with_name_and_type.type = new DataTypeUInt64();
	ColumnUInt64 * column = new ColumnUInt64(block_size);
	ColumnUInt64::Container_t & vec = column->getData();
	column_with_name_and_type.column = column;

	size_t curr = next;		/// Локальная переменная почему-то работает быстрее (>20%), чем член класса.
	UInt64 * pos = &vec[0];	/// Это тоже ускоряет код.
	UInt64 * end = &vec[block_size];
	while (pos < end)
		*pos++ = curr++;
	next = curr;

	res.insert(column_with_name_and_type);

	return res;
}
