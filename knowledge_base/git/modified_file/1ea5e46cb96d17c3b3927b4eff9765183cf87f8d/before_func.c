	for (i = 0; i < nr_objects; i++)
		hashwrite_be32(f, pack_order[i]);
}

static void write_rev_trailer(struct hashfile *f, const unsigned char *hash)
{
	hashwrite(f, hash, the_hash_algo->rawsz);
}

const char *write_rev_file(const char *rev_name,
			   struct pack_idx_entry **objects,
			   uint32_t nr_objects,
			   const unsigned char *hash,
			   unsigned flags)
{
	uint32_t *pack_order;
	uint32_t i;
	const char *ret;

	ALLOC_ARRAY(pack_order, nr_objects);
	for (i = 0; i < nr_objects; i++)
		pack_order[i] = i;
