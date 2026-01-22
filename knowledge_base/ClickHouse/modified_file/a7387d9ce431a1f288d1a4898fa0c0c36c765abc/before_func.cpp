

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

		if (col_to.empty())
			to.getByPosition(column_no).column = col_from.clone();
		else
			for (size_t row_no = 0; row_no < rows; ++row_no)
				col_to.insertFrom(col_from, row_no);
