{
	/// Выбрасывает N последних столбцов из in (если их меньше, то все) и кладет в result их комбинацию.
	static void execute(UInt8ColumnPtrs & in, UInt8Container & result)
	{
		if (N > in.size()){
			AssociativeOperationImpl<Op, N - 1>::execute(in, result);
			return;
		}

		AssociativeOperationImpl<Op, N> operation(in);
		in.erase(in.end() - N, in.end());

		size_t n = result.size();
		for (size_t i = 0; i < n; ++i)
		{
			result[i] = operation.apply(i);
