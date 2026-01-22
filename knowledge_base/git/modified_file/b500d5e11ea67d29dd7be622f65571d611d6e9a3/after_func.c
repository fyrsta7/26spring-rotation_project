	struct object_entry_pool *o;

	/* Build the table of object IDs. */
	idx = xmalloc(object_count * sizeof(*idx));
	c = idx;
	for (o = blocks; o; o = o->next_pool)
		for (e = o->next_free; e-- != o->entries;)
			if (pack_id == e->pack_id)
				*c++ = &e->idx;
	last = idx + object_count;
	if (c != last)
		die("internal consistency error creating the index");

	tmpfile = write_idx_file(NULL, idx, object_count, pack_data->sha1);
	free(idx);
	return tmpfile;
}

static char *keep_pack(const char *curr_index_name)
{
	static char name[PATH_MAX];
	static const char *keep_msg = "fast-import";
	int keep_fd;

	keep_fd = odb_pack_keep(name, sizeof(name), pack_data->sha1);
	if (keep_fd < 0)
		die_errno("cannot create keep file");
	write_or_die(keep_fd, keep_msg, strlen(keep_msg));
	if (close(keep_fd))
		die_errno("failed to write keep file");

	snprintf(name, sizeof(name), "%s/pack/pack-%s.pack",
		 get_object_directory(), sha1_to_hex(pack_data->sha1));
	if (move_temp_to_file(pack_data->pack_name, name))
		die("cannot store pack file");

	snprintf(name, sizeof(name), "%s/pack/pack-%s.idx",
		 get_object_directory(), sha1_to_hex(pack_data->sha1));
	if (move_temp_to_file(curr_index_name, name))
		die("cannot store index file");
	free((void *)curr_index_name);
	return name;
}

static void unkeep_all_packs(void)
{
	static char name[PATH_MAX];
	int k;

	for (k = 0; k < pack_id; k++) {
		struct packed_git *p = all_packs[k];
		snprintf(name, sizeof(name), "%s/pack/pack-%s.keep",
			 get_object_directory(), sha1_to_hex(p->sha1));
		unlink_or_warn(name);
	}
}

static void end_packfile(void)
{
	struct packed_git *old_p = pack_data, *new_p;

	clear_delta_base_cache();
	if (object_count) {
		unsigned char cur_pack_sha1[20];
		char *idx_name;
		int i;
		struct branch *b;
		struct tag *t;

		close_pack_windows(pack_data);
		sha1close(pack_file, cur_pack_sha1, 0);
		fixup_pack_header_footer(pack_data->pack_fd, pack_data->sha1,
				    pack_data->pack_name, object_count,
				    cur_pack_sha1, pack_size);
		close(pack_data->pack_fd);
		idx_name = keep_pack(create_index());

		/* Register the packfile with core git's machinery. */
		new_p = add_packed_git(idx_name, strlen(idx_name), 1);
		if (!new_p)
			die("core git rejected index %s", idx_name);
		all_packs[pack_id] = new_p;
		install_packed_git(new_p);

		/* Print the boundary */
		if (pack_edges) {
			fprintf(pack_edges, "%s:", new_p->pack_name);
			for (i = 0; i < branch_table_sz; i++) {
				for (b = branch_table[i]; b; b = b->table_next_branch) {
					if (b->pack_id == pack_id)
						fprintf(pack_edges, " %s", sha1_to_hex(b->sha1));
				}
			}
			for (t = first_tag; t; t = t->next_tag) {
				if (t->pack_id == pack_id)
					fprintf(pack_edges, " %s", sha1_to_hex(t->sha1));
			}
			fputc('\n', pack_edges);
			fflush(pack_edges);
		}

		pack_id++;
	}
	else {
		close(old_p->pack_fd);
		unlink_or_warn(old_p->pack_name);
	}
	free(old_p);

	/* We can't carry a delta across packfiles. */
	strbuf_release(&last_blob.data);
	last_blob.offset = 0;
	last_blob.depth = 0;
}

static void cycle_packfile(void)
{
	end_packfile();
	start_packfile();
}

static size_t encode_header(
	enum object_type type,
	uintmax_t size,
	unsigned char *hdr)
{
	int n = 1;
	unsigned char c;

	if (type < OBJ_COMMIT || type > OBJ_REF_DELTA)
		die("bad type %d", type);

	c = (type << 4) | (size & 15);
	size >>= 4;
