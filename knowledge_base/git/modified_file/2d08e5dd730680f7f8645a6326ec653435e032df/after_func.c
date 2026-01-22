	 * Make sure none of the hash buckets has more entries than
	 * we're willing to test.  Otherwise we cull the entry list
	 * uniformly to still preserve a good repartition across
	 * the reference buffer.
	 */
	for (i = 0; i < hsize; i++) {
		if (hash_count[i] < HASH_LIMIT)
			continue;
		entry = hash[i];
		do {
			struct index_entry *keep = entry;
			int skip = hash_count[i] / HASH_LIMIT / 2;
			do {
				entry = entry->next;
			} while(--skip && entry);
			keep->next = entry;
		} while(entry);
	}
	free(hash_count);

	return index;
}

void free_delta_index(struct delta_index *index)
{
	free(index);
}

/*
 * The maximum size for any opcode sequence, including the initial header
 * plus rabin window plus biggest copy.
 */
#define MAX_OP_SIZE	(5 + 5 + 1 + RABIN_WINDOW + 7)

void *
create_delta(const struct delta_index *index,
	     const void *trg_buf, unsigned long trg_size,
	     unsigned long *delta_size, unsigned long max_size)
{
	unsigned int i, outpos, outsize, val;
	int inscnt;
	const unsigned char *ref_data, *ref_top, *data, *top;
	unsigned char *out;

	if (!trg_buf || !trg_size)
		return NULL;

	outpos = 0;
	outsize = 8192;
	if (max_size && outsize >= max_size)
		outsize = max_size + MAX_OP_SIZE + 1;
	out = malloc(outsize);
	if (!out)
		return NULL;

	/* store reference buffer size */
	i = index->src_size;
	while (i >= 0x80) {
		out[outpos++] = i | 0x80;
		i >>= 7;
	}
	out[outpos++] = i;

	/* store target buffer size */
	i = trg_size;
	while (i >= 0x80) {
		out[outpos++] = i | 0x80;
		i >>= 7;
	}
	out[outpos++] = i;

	ref_data = index->src_buf;
	ref_top = ref_data + index->src_size;
	data = trg_buf;
	top = trg_buf + trg_size;

	outpos++;
	val = 0;
	for (i = 0; i < RABIN_WINDOW && data < top; i++, data++) {
		out[outpos++] = *data;
		val = ((val << 8) | *data) ^ T[val >> RABIN_SHIFT];
	}
	inscnt = i;

	while (data < top) {
		unsigned int moff = 0, msize = 0;
		struct index_entry *entry;
		val ^= U[data[-RABIN_WINDOW]];
		val = ((val << 8) | *data) ^ T[val >> RABIN_SHIFT];
		i = val & index->hash_mask;
		for (entry = index->hash[i]; entry; entry = entry->next) {
			const unsigned char *ref = entry->ptr;
			const unsigned char *src = data;
			unsigned int ref_size = ref_top - ref;
			if (entry->val != val)
				continue;
			if (ref_size > top - src)
				ref_size = top - src;
			if (ref_size > 0x10000)
				ref_size = 0x10000;
			if (ref_size <= msize)
				break;
			while (ref_size-- && *src++ == *ref)
				ref++;
			if (msize < ref - entry->ptr) {
				/* this is our best match so far */
				msize = ref - entry->ptr;
				moff = entry->ptr - ref_data;
			}
		}

		if (msize < 4) {
			if (!inscnt)
				outpos++;
			out[outpos++] = *data++;
			inscnt++;
			if (inscnt == 0x7f) {
				out[outpos - inscnt - 1] = inscnt;
				inscnt = 0;
			}
		} else {
			unsigned char *op;

			if (msize >= RABIN_WINDOW) {
				const unsigned char *sk;
				sk = data + msize - RABIN_WINDOW;
				val = 0;
				for (i = 0; i < RABIN_WINDOW; i++)
					val = ((val << 8) | *sk++) ^ T[val >> RABIN_SHIFT];
			} else {
				const unsigned char *sk = data + 1;
				for (i = 1; i < msize; i++) {
					val ^= U[sk[-RABIN_WINDOW]];
					val = ((val << 8) | *sk++) ^ T[val >> RABIN_SHIFT];
				}
			}

			if (inscnt) {
				while (moff && ref_data[moff-1] == data[-1]) {
					if (msize == 0x10000)
						break;
					/* we can match one byte back */
					msize++;
					moff--;
					data--;
					outpos--;
					if (--inscnt)
						continue;
					outpos--;  /* remove count slot */
					inscnt--;  /* make it -1 */
					break;
				}
				out[outpos - inscnt - 1] = inscnt;
				inscnt = 0;
			}

			data += msize;
			op = out + outpos++;
			i = 0x80;

			if (moff & 0xff) { out[outpos++] = moff; i |= 0x01; }
			moff >>= 8;
			if (moff & 0xff) { out[outpos++] = moff; i |= 0x02; }
			moff >>= 8;
			if (moff & 0xff) { out[outpos++] = moff; i |= 0x04; }
			moff >>= 8;
			if (moff & 0xff) { out[outpos++] = moff; i |= 0x08; }
