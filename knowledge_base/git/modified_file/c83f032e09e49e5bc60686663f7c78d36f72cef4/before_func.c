			entry->size = get_size_from_delta(p, &w_curs,
					entry->in_pack_offset + entry->in_pack_header_size);
			unuse_pack(&w_curs);
			return;
		}

		/*
		 * No choice but to fall back to the recursive delta walk
		 * with sha1_object_info() to find about the object type
		 * at this point...
		 */
		unuse_pack(&w_curs);
	}

	entry->type = sha1_object_info(entry->idx.sha1, &entry->size);
	if (entry->type < 0)
		die("unable to get type of object %s",
		    sha1_to_hex(entry->idx.sha1));
}

static int pack_offset_sort(const void *_a, const void *_b)
{
	const struct object_entry *a = *(struct object_entry **)_a;
	const struct object_entry *b = *(struct object_entry **)_b;

	/* avoid filesystem trashing with loose objects */
	if (!a->in_pack && !b->in_pack)
		return hashcmp(a->idx.sha1, b->idx.sha1);

	if (a->in_pack < b->in_pack)
		return -1;
	if (a->in_pack > b->in_pack)
		return 1;
	return a->in_pack_offset < b->in_pack_offset ? -1 :
			(a->in_pack_offset > b->in_pack_offset);
}

static void get_object_details(void)
{
	uint32_t i;
	struct object_entry **sorted_by_offset;

	sorted_by_offset = xcalloc(nr_objects, sizeof(struct object_entry *));
	for (i = 0; i < nr_objects; i++)
		sorted_by_offset[i] = objects + i;
	qsort(sorted_by_offset, nr_objects, sizeof(*sorted_by_offset), pack_offset_sort);

	prepare_pack_ix();
	for (i = 0; i < nr_objects; i++)
		check_object(sorted_by_offset[i]);
	free(sorted_by_offset);
}

static int type_size_sort(const void *_a, const void *_b)
{
	const struct object_entry *a = *(struct object_entry **)_a;
	const struct object_entry *b = *(struct object_entry **)_b;

	if (a->type < b->type)
		return -1;
	if (a->type > b->type)
		return 1;
	if (a->hash < b->hash)
		return -1;
	if (a->hash > b->hash)
		return 1;
	if (a->preferred_base < b->preferred_base)
		return -1;
	if (a->preferred_base > b->preferred_base)
		return 1;
	if (a->size < b->size)
		return -1;
	if (a->size > b->size)
		return 1;
	return a > b ? -1 : (a < b);  /* newest last */
}

struct unpacked {
	struct object_entry *entry;
	void *data;
	struct delta_index *index;
};

static int delta_cacheable(struct unpacked *trg, struct unpacked *src,
			    unsigned long src_size, unsigned long trg_size,
			    unsigned long delta_size)
{
	if (max_delta_cache_size && delta_cache_size + delta_size > max_delta_cache_size)
		return 0;

