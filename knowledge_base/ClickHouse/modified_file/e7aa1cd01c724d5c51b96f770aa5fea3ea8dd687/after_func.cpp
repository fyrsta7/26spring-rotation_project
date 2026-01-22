template <int UNROLL_TIMES>
static NO_INLINE void deserializeBinarySSE2(ColumnString::Chars_t & data, ColumnString::Offsets_t & offsets, ReadBuffer & istr, size_t limit)
{
	size_t offset = data.size();
	for (size_t i = 0; i < limit; ++i)
	{
		if (istr.eof())
			break;

		UInt64 size;
		readVarUInt(size, istr);

		offset += size + 1;
		offsets.push_back(offset);

		data.resize(offset);

		if (size)
		{
			/// Оптимистичная ветка, в которой возможно более эффективное копирование.
			if (offset + 16 * UNROLL_TIMES <= data.capacity() && istr.position() + size + 16 * UNROLL_TIMES <= istr.buffer().end())
			{
				const __m128i * sse_src_pos = reinterpret_cast<const __m128i *>(istr.position());
				const __m128i * sse_src_end = sse_src_pos + (size + (16 * UNROLL_TIMES - 1)) / (16 * UNROLL_TIMES);
				__m128i * sse_dst_pos = reinterpret_cast<__m128i *>(&data[offset - size - 1]);

				while (sse_src_pos < sse_src_end)
				{
					/// NOTE gcc 4.9.2 разворачивает цикл, но почему-то использует только один xmm регистр.
					///for (size_t j = 0; j < UNROLL_TIMES; ++j)
					///	_mm_storeu_si128(sse_dst_pos + j, _mm_loadu_si128(sse_src_pos + j));

					sse_src_pos += UNROLL_TIMES;
					sse_dst_pos += UNROLL_TIMES;

					if (UNROLL_TIMES >= 3) __asm__("movdqu %0, %%xmm0" :: "m"(sse_src_pos[-3]));
					if (UNROLL_TIMES >= 2) __asm__("movdqu %0, %%xmm1" :: "m"(sse_src_pos[-2]));
					if (UNROLL_TIMES >= 1) __asm__("movdqu %0, %%xmm2" :: "m"(sse_src_pos[-1]));
					if (UNROLL_TIMES >= 0) __asm__("movdqu %0, %%xmm3" :: "m"(sse_src_pos[0]));

					if (UNROLL_TIMES >= 3) __asm__("movdqu %%xmm0, %0" : "=m"(sse_dst_pos[-3]));
					if (UNROLL_TIMES >= 2) __asm__("movdqu %%xmm1, %0" : "=m"(sse_dst_pos[-2]));
					if (UNROLL_TIMES >= 1) __asm__("movdqu %%xmm2, %0" : "=m"(sse_dst_pos[-1]));
					if (UNROLL_TIMES >= 0) __asm__("movdqu %%xmm3, %0" : "=m"(sse_dst_pos[0]));
				}

				istr.position() += size;
			}
			else
			{
				istr.readStrict(reinterpret_cast<char*>(&data[offset - size - 1]), size);
			}
		}

		data[offset - 1] = 0;
	}
}
