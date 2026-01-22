

static void appendBlock(const Block & from, Block & to)
{
	size_t rows = from.rows();
	for (size_t column_no = 0, columns = to.columns(); column_no < columns; ++column_no)
	{
		const IColumn & col_from = *from.getByPosition(column_no).column.get();
		IColumn & col_to = *to.getByPosition(column_no).column.get();

		if (col_from.getName() != col_to.getName())
			throw Exception("Cannot append block to another: different type of columns at index " + toString(column_no)
				+ ". Block 1: " + from.dumpStructure() + ". Block 2: " + to.dumpStructure(), ErrorCodes::BLOCKS_HAS_DIFFERENT_STRUCTURE);

		col_to.insertRangeFrom(col_from, 0, rows);
