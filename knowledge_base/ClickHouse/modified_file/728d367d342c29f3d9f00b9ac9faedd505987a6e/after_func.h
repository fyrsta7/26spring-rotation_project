	void execute(Block & block, const ColumnNumbers & arguments, const size_t result) override
	{
		block.getByPosition(result).column = new ColumnConst<Float64>{
			block.rowsInFirstColumn(),
			Impl::value
		};
	}
