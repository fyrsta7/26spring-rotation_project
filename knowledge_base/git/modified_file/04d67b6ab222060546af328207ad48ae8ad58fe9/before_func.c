{
	struct string_list *list = (struct string_list *)cb_data;

	string_list_append(list, oid_to_hex(oid));
	return 0;
}

void write_commit_graph_reachable(const char *obj_dir, int append,
				  int report_progress)
{
	struct string_list list = STRING_LIST_INIT_DUP;

	for_each_ref(add_ref_to_list, &list);
	write_commit_graph(obj_dir, NULL, &list, append, report_progress);

	string_list_clear(&list, 0);
}

void write_commit_graph(const char *obj_dir,
			struct string_list *pack_indexes,
			struct string_list *commit_hex,
			int append, int report_progress)
{
	struct packed_oid_list oids;
	struct packed_commit_list commits;
	struct hashfile *f;
	uint32_t i, count_distinct = 0;
	char *graph_name;
	struct lock_file lk = LOCK_INIT;
	uint32_t chunk_ids[5];
	uint64_t chunk_offsets[5];
	int num_chunks;
	int num_extra_edges;
	struct commit_list *parent;
	struct progress *progress = NULL;
	const unsigned hashsz = the_hash_algo->rawsz;

	if (!commit_graph_compatible(the_repository))
		return;

	oids.nr = 0;
	oids.alloc = approximate_object_count() / 32;
	oids.progress = NULL;
	oids.progress_done = 0;

	if (append) {
		prepare_commit_graph_one(the_repository, obj_dir);
		if (the_repository->objects->commit_graph)
			oids.alloc += the_repository->objects->commit_graph->num_commits;
	}

	if (oids.alloc < 1024)
		oids.alloc = 1024;
	ALLOC_ARRAY(oids.list, oids.alloc);

	if (append && the_repository->objects->commit_graph) {
		struct commit_graph *commit_graph =
			the_repository->objects->commit_graph;
		for (i = 0; i < commit_graph->num_commits; i++) {
			const unsigned char *hash = commit_graph->chunk_oid_lookup +
				commit_graph->hash_len * i;
			hashcpy(oids.list[oids.nr++].hash, hash);
		}
	}

	if (pack_indexes) {
		struct strbuf packname = STRBUF_INIT;
		int dirlen;
		strbuf_addf(&packname, "%s/pack/", obj_dir);
		dirlen = packname.len;
		if (report_progress) {
			oids.progress = start_delayed_progress(
				_("Finding commits for commit graph"), 0);
			oids.progress_done = 0;
		}
		for (i = 0; i < pack_indexes->nr; i++) {
			struct packed_git *p;
			strbuf_setlen(&packname, dirlen);
			strbuf_addstr(&packname, pack_indexes->items[i].string);
			p = add_packed_git(packname.buf, packname.len, 1);
			if (!p)
				die(_("error adding pack %s"), packname.buf);
			if (open_pack_index(p))
				die(_("error opening index for %s"), packname.buf);
			for_each_object_in_pack(p, add_packed_commits, &oids, 0);
			close_pack(p);
			free(p);
		}
		stop_progress(&oids.progress);
		strbuf_release(&packname);
	}

	if (commit_hex) {
		if (report_progress)
			progress = start_delayed_progress(
				_("Finding commits for commit graph"),
				commit_hex->nr);
		for (i = 0; i < commit_hex->nr; i++) {
			const char *end;
			struct object_id oid;
			struct commit *result;

			display_progress(progress, i + 1);
			if (commit_hex->items[i].string &&
			    parse_oid_hex(commit_hex->items[i].string, &oid, &end))
				continue;

			result = lookup_commit_reference_gently(the_repository, &oid, 1);

			if (result) {
				ALLOC_GROW(oids.list, oids.nr + 1, oids.alloc);
				oidcpy(&oids.list[oids.nr], &(result->object.oid));
				oids.nr++;
			}
		}
		stop_progress(&progress);
	}

	if (!pack_indexes && !commit_hex) {
		if (report_progress)
			oids.progress = start_delayed_progress(
				_("Finding commits for commit graph"), 0);
		for_each_packed_object(add_packed_commits, &oids, 0);
		stop_progress(&oids.progress);
	}

	close_reachable(&oids, report_progress);

	QSORT(oids.list, oids.nr, commit_compare);

	count_distinct = 1;
	for (i = 1; i < oids.nr; i++) {
		if (!oideq(&oids.list[i - 1], &oids.list[i]))
			count_distinct++;
	}

	if (count_distinct >= GRAPH_EDGE_LAST_MASK)
		die(_("the commit graph format cannot write %d commits"), count_distinct);

	commits.nr = 0;
	commits.alloc = count_distinct;
	ALLOC_ARRAY(commits.list, commits.alloc);

	num_extra_edges = 0;
	for (i = 0; i < oids.nr; i++) {
		int num_parents = 0;
		if (i > 0 && oideq(&oids.list[i - 1], &oids.list[i]))
			continue;

		commits.list[commits.nr] = lookup_commit(the_repository, &oids.list[i]);
		parse_commit(commits.list[commits.nr]);

		for (parent = commits.list[commits.nr]->parents;
		     parent; parent = parent->next)
			num_parents++;

		if (num_parents > 2)
			num_extra_edges += num_parents - 1;

		commits.nr++;
	}
	num_chunks = num_extra_edges ? 4 : 3;

	if (commits.nr >= GRAPH_EDGE_LAST_MASK)
		die(_("too many commits to write graph"));

	compute_generation_numbers(&commits, report_progress);

	graph_name = get_commit_graph_filename(obj_dir);
	if (safe_create_leading_directories(graph_name)) {
		UNLEAK(graph_name);
		die_errno(_("unable to create leading directories of %s"),
			  graph_name);
	}

	hold_lock_file_for_update(&lk, graph_name, LOCK_DIE_ON_ERROR);
	f = hashfd(lk.tempfile->fd, lk.tempfile->filename.buf);

	hashwrite_be32(f, GRAPH_SIGNATURE);

	hashwrite_u8(f, GRAPH_VERSION);
	hashwrite_u8(f, oid_version());
	hashwrite_u8(f, num_chunks);
	hashwrite_u8(f, 0); /* unused padding byte */

	chunk_ids[0] = GRAPH_CHUNKID_OIDFANOUT;
	chunk_ids[1] = GRAPH_CHUNKID_OIDLOOKUP;
	chunk_ids[2] = GRAPH_CHUNKID_DATA;
	if (num_extra_edges)
		chunk_ids[3] = GRAPH_CHUNKID_LARGEEDGES;
	else
		chunk_ids[3] = 0;
	chunk_ids[4] = 0;

	chunk_offsets[0] = 8 + (num_chunks + 1) * GRAPH_CHUNKLOOKUP_WIDTH;
	chunk_offsets[1] = chunk_offsets[0] + GRAPH_FANOUT_SIZE;
	chunk_offsets[2] = chunk_offsets[1] + hashsz * commits.nr;
	chunk_offsets[3] = chunk_offsets[2] + (hashsz + 16) * commits.nr;
	chunk_offsets[4] = chunk_offsets[3] + 4 * num_extra_edges;

	for (i = 0; i <= num_chunks; i++) {
		uint32_t chunk_write[3];

		chunk_write[0] = htonl(chunk_ids[i]);
