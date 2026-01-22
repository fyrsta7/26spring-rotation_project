	result = strbuf_cmp(&a->key, &rkey);
	strbuf_release(&rkey);
	return result;
}

void block_iter_copy_from(struct block_iter *dest, struct block_iter *src)
{
	dest->br = src->br;
	dest->next_off = src->next_off;
	strbuf_reset(&dest->last_key);
	strbuf_addbuf(&dest->last_key, &src->last_key);
}

int block_iter_next(struct block_iter *it, struct reftable_record *rec)
{
	struct string_view in = {
		.buf = it->br->block.data + it->next_off,
		.len = it->br->block_len - it->next_off,
	};
	struct string_view start = in;
	uint8_t extra = 0;
	int n = 0;

	if (it->next_off >= it->br->block_len)
		return 1;

	n = reftable_decode_key(&it->key, &extra, it->last_key, in);
	if (n < 0)
		return -1;

	if (!it->key.len)
