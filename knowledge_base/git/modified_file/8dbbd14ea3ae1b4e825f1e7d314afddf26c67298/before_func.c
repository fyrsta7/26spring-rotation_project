		 * Depth of objects that depend on the entry -- this
		 * is subtracted from depth-max to break too deep
		 * delta chain because of delta data reusing.
		 * However, we loosen this restriction when we know we
		 * are creating a thin pack -- it will have to be
		 * expanded on the other end anyway, so do not
		 * artificially cut the delta chain and let it go as
		 * deep as it wants.
		 */
		for (i = 0, entry = objects; i < nr_objects; i++, entry++)
			if (!entry->delta && entry->delta_child)
				entry->delta_limit =
					check_delta_limit(entry, 1);
	}
}

typedef int (*entry_sort_t)(const struct object_entry *, const struct object_entry *);

static entry_sort_t current_sort;

static int sort_comparator(const void *_a, const void *_b)
{
	struct object_entry *a = *(struct object_entry **)_a;
	struct object_entry *b = *(struct object_entry **)_b;
	return current_sort(a,b);
}

static struct object_entry **create_sorted_list(entry_sort_t sort)
{
	struct object_entry **list = xmalloc(nr_objects * sizeof(struct object_entry *));
	int i;

	for (i = 0; i < nr_objects; i++)
		list[i] = objects + i;
	current_sort = sort;
	qsort(list, nr_objects, sizeof(struct object_entry *), sort_comparator);
	return list;
}

static int sha1_sort(const struct object_entry *a, const struct object_entry *b)
{
	return memcmp(a->sha1, b->sha1, 20);
}

static struct object_entry **create_final_object_list(void)
{
	struct object_entry **list;
	int i, j;

	for (i = nr_result = 0; i < nr_objects; i++)
		if (!objects[i].preferred_base)
			nr_result++;
