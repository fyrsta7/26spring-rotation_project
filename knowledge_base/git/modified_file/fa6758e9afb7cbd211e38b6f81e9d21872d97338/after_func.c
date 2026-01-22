
	/* Build the table of object IDs. */
	ALLOC_ARRAY(idx, object_count);
	c = idx;
	for (o = blocks; o; o = o->next_pool)
		for (e = o->next_free; e-- != o->entries;)
			if (pack_id == e->pack_id)
				*c++ = &e->idx;
	last = idx + object_count;
	if (c != last)
		die("internal consistency error creating the index");

	tmpfile = write_idx_file(NULL, idx, object_count, &pack_idx_opts, pack_data->sha1);
	free(idx);
	return tmpfile;
}

static char *keep_pack(const char *curr_index_name)
{
	static const char *keep_msg = "fast-import";
	struct strbuf name = STRBUF_INIT;
	int keep_fd;

	odb_pack_name(&name, pack_data->sha1, "keep");
	keep_fd = odb_pack_keep(name.buf);
	if (keep_fd < 0)
		die_errno("cannot create keep file");
	write_or_die(keep_fd, keep_msg, strlen(keep_msg));
	if (close(keep_fd))
		die_errno("failed to write keep file");

	odb_pack_name(&name, pack_data->sha1, "pack");
	if (finalize_object_file(pack_data->pack_name, name.buf))
		die("cannot store pack file");

	odb_pack_name(&name, pack_data->sha1, "idx");
	if (finalize_object_file(curr_index_name, name.buf))
		die("cannot store index file");
	free((void *)curr_index_name);
	return strbuf_detach(&name, NULL);
}

static void unkeep_all_packs(void)
{
	struct strbuf name = STRBUF_INIT;
	int k;

	for (k = 0; k < pack_id; k++) {
		struct packed_git *p = all_packs[k];
		odb_pack_name(&name, p->sha1, "keep");
		unlink_or_warn(name.buf);
	}
	strbuf_release(&name);
}

static int loosen_small_pack(const struct packed_git *p)
{
	struct child_process unpack = CHILD_PROCESS_INIT;

	if (lseek(p->pack_fd, 0, SEEK_SET) < 0)
		die_errno("Failed seeking to start of '%s'", p->pack_name);

	unpack.in = p->pack_fd;
	unpack.git_cmd = 1;
	unpack.stdout_to_stderr = 1;
	argv_array_push(&unpack.args, "unpack-objects");
	if (!show_stats)
		argv_array_push(&unpack.args, "-q");

	return run_command(&unpack);
}

static void end_packfile(void)
{
	static int running;

	if (running || !pack_data)
		return;

	running = 1;
	clear_delta_base_cache();
	if (object_count) {
		struct packed_git *new_p;
		struct object_id cur_pack_oid;
		char *idx_name;
		int i;
		struct branch *b;
		struct tag *t;

		close_pack_windows(pack_data);
		finalize_hashfile(pack_file, cur_pack_oid.hash, 0);
		fixup_pack_header_footer(pack_data->pack_fd, pack_data->sha1,
				    pack_data->pack_name, object_count,
				    cur_pack_oid.hash, pack_size);

		if (object_count <= unpack_limit) {
			if (!loosen_small_pack(pack_data)) {
				invalidate_pack_id(pack_id);
				goto discard_pack;
			}
		}

		close(pack_data->pack_fd);
		idx_name = keep_pack(create_index());

		/* Register the packfile with core git's machinery. */
		new_p = add_packed_git(idx_name, strlen(idx_name), 1);
		if (!new_p)
			die("core git rejected index %s", idx_name);
		all_packs[pack_id] = new_p;
		install_packed_git(the_repository, new_p);
		free(idx_name);

		/* Print the boundary */
		if (pack_edges) {
			fprintf(pack_edges, "%s:", new_p->pack_name);
			for (i = 0; i < branch_table_sz; i++) {
				for (b = branch_table[i]; b; b = b->table_next_branch) {
					if (b->pack_id == pack_id)
						fprintf(pack_edges, " %s",
							oid_to_hex(&b->oid));
				}
			}
			for (t = first_tag; t; t = t->next_tag) {
				if (t->pack_id == pack_id)
					fprintf(pack_edges, " %s",
						oid_to_hex(&t->oid));
			}
			fputc('\n', pack_edges);
			fflush(pack_edges);
		}

		pack_id++;
	}
	else {
discard_pack:
		close(pack_data->pack_fd);
