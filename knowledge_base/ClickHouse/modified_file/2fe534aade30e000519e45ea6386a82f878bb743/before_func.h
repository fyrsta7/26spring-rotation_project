			+ toString(start) + ", length = "
			+ toString(length) + " are out of bound in IColumnVector<T>::cut() method"
			" (data.size() = " + toString(data.size()) + ").",
							ErrorCodes::PARAMETER_OUT_OF_BOUND);

			Self * res = new Self(length);
		memcpy(&res->getData()[0], &data[start], length * sizeof(data[0]));
		return res;
	}

	ColumnPtr filter(const IColumn::Filter & filt) const
	{
		size_t size = data.size();
		if (size != filt.size())
			throw Exception("Size of filter doesn't match size of column.", ErrorCodes::SIZES_OF_COLUMNS_DOESNT_MATCH);

		Self * res_ = new Self;
		ColumnPtr res = res_;
		typename Self::Container_t & res_data = res_->getData();
		res_data.reserve(size);

		/** Чуть более оптимизированная версия.
		  * Исходит из допущения, что часто куски последовательно идущих значений
		  *  полностью проходят или полностью не проходят фильтр.
		  * Поэтому, будем оптимистично проверять куски по 16 значений.
		  */
		const UInt8 * filt_pos = &filt[0];
		const UInt8 * filt_end = filt_pos + size;
		const UInt8 * filt_end_sse = filt_pos + size / 16 * 16;
		const T * data_pos = &data[0];

		while (filt_pos < filt_end_sse)
		{
			int mask = _mm_movemask_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(filt_pos)));

			if (0 == mask)
			{
				/// Ничего не вставляем.
			}
			else if (0xFFFF == mask)
			{
				res_data.insert_assume_reserved(data_pos, data_pos + 16);
			}
			else
			{
				for (size_t i = 0; i < 16; ++i)
					if (filt_pos[i])
						res_data.push_back(data_pos[i]);
			}

			filt_pos += 16;
