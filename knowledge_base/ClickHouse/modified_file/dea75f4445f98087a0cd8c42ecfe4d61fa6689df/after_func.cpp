			if (data_types[i]->getName() != block.getByPosition(arguments[i]).type->getName())
				throw Exception("Types of column " + toString(i + 1) + " in section IN don't match: " + data_types[i]->getName() + " on the right, " + block.getByPosition(arguments[i]).type->getName() + " on the left.", ErrorCodes::TYPE_MISMATCH);
		}

		executeOrdinary(key_columns, vec_res, negative);
	}
}

void Set::executeOrdinary(const ConstColumnPlainPtrs & key_columns, ColumnUInt8::Container_t & vec_res, bool negative) const
{
	size_t keys_size = data_types.size();
	size_t rows = key_columns[0]->size();
	Row key(keys_size);

	if (type == KEY_64)
	{
		const SetUInt64 & set = *key64;
		const IColumn & column = *key_columns[0];

		/// Для всех строчек
		for (size_t i = 0; i < rows; ++i)
		{
			/// Строим ключ
			UInt64 key = column.get64(i);
			vec_res[i] = negative ^ (set.end() != set.find(key));
		}
	}
	else if (type == KEY_STRING)
	{
		const SetString & set = *key_string;
		const IColumn & column = *key_columns[0];

		if (const ColumnString * column_string = typeid_cast<const ColumnString *>(&column))
		{
			const ColumnString::Offsets_t & offsets = column_string->getOffsets();
			const ColumnString::Chars_t & data = column_string->getChars();

			StringRef prev_key;
			bool prev_result;

			/// Для всех строчек
			for (size_t i = 0; i < rows; ++i)
			{
				/// Строим ключ
				StringRef ref(&data[i == 0 ? 0 : offsets[i - 1]], (i == 0 ? offsets[i] : (offsets[i] - offsets[i - 1])) - 1);

				if (i != 0 && ref == prev_key)
					vec_res[i] = prev_result;
				else
				{
					prev_result = negative ^ (set.end() != set.find(ref));
					prev_key = ref;
					vec_res[i] = prev_result;
				}
			}
		}
		else if (const ColumnFixedString * column_string = typeid_cast<const ColumnFixedString *>(&column))
		{
			size_t n = column_string->getN();
			const ColumnFixedString::Chars_t & data = column_string->getChars();

			/// Для всех строчек
			for (size_t i = 0; i < rows; ++i)
			{
				/// Строим ключ
				StringRef ref(&data[i * n], n);
				vec_res[i] = negative ^ (set.end() != set.find(ref));
			}
		}
		else if (const ColumnConstString * column_string = typeid_cast<const ColumnConstString *>(&column))
		{
			bool res = negative ^ (set.end() != set.find(StringRef(column_string->getData())));

			/// Для всех строчек
			for (size_t i = 0; i < rows; ++i)
				vec_res[i] = res;
		}
		else
			throw Exception("Illegal type of column when creating set with string key: " + column.getName(), ErrorCodes::ILLEGAL_COLUMN);
	}
	else if (type == HASHED)
	{
		const SetHashed & set = *hashed;
