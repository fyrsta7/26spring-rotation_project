
void init_bloom_filters(void)
{
	init_bloom_filter_slab(&bloom_filters);
}

struct bloom_filter *get_bloom_filter(struct repository *r,
				      struct commit *c,
					  int compute_if_not_present)
{
	struct bloom_filter *filter;
	struct bloom_filter_settings settings = DEFAULT_BLOOM_FILTER_SETTINGS;
	int i;
	struct diff_options diffopt;
	int max_changes = 512;

	if (bloom_filters.slab_size == 0)
		return NULL;

	filter = bloom_filter_slab_at(&bloom_filters, c);

	if (!filter->data) {
		load_commit_graph_info(r, c);
		if (c->graph_pos != COMMIT_NOT_FROM_GRAPH &&
			r->objects->commit_graph->chunk_bloom_indexes) {
			if (load_bloom_filter_from_graph(r->objects->commit_graph, filter, c))
				return filter;
			else
				return NULL;
		}
	}

	if (filter->data || !compute_if_not_present)
		return filter;

	repo_diff_setup(r, &diffopt);
	diffopt.flags.recursive = 1;
	diffopt.max_changes = max_changes;
	diff_setup_done(&diffopt);

	if (c->parents)
		diff_tree_oid(&c->parents->item->object.oid, &c->object.oid, "", &diffopt);
	else
		diff_tree_oid(NULL, &c->object.oid, "", &diffopt);
	diffcore_std(&diffopt);

	if (diff_queued_diff.nr <= max_changes) {
		struct hashmap pathmap;
		struct pathmap_hash_entry *e;
		struct hashmap_iter iter;
		hashmap_init(&pathmap, NULL, NULL, 0);

		for (i = 0; i < diff_queued_diff.nr; i++) {
			const char *path = diff_queued_diff.queue[i]->two->path;

			/*
			* Add each leading directory of the changed file, i.e. for
			* 'dir/subdir/file' add 'dir' and 'dir/subdir' as well, so
			* the Bloom filter could be used to speed up commands like
			* 'git log dir/subdir', too.
			*
			* Note that directories are added without the trailing '/'.
			*/
			do {
				char *last_slash = strrchr(path, '/');

				FLEX_ALLOC_STR(e, path, path);
				hashmap_entry_init(&e->entry, strhash(path));
				hashmap_add(&pathmap, &e->entry);

				if (!last_slash)
					last_slash = (char*)path;
				*last_slash = '\0';

			} while (*path);

			diff_free_filepair(diff_queued_diff.queue[i]);
		}

		filter->len = (hashmap_get_size(&pathmap) * settings.bits_per_entry + BITS_PER_WORD - 1) / BITS_PER_WORD;
		filter->data = xcalloc(filter->len, sizeof(unsigned char));

		hashmap_for_each_entry(&pathmap, &iter, e, entry) {
			struct bloom_key key;
			fill_bloom_key(e->path, strlen(e->path), &key, &settings);
			add_key_to_filter(&key, filter, &settings);
		}

