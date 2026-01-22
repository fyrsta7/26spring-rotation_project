	b = lookup_branch(sp);
	if (!b)
		b = new_branch(sp);

	read_next_command();
	parse_mark();
	if (!prefixcmp(command_buf.buf, "author ")) {
		author = parse_ident(command_buf.buf + 7);
		read_next_command();
	}
	if (!prefixcmp(command_buf.buf, "committer ")) {
		committer = parse_ident(command_buf.buf + 10);
		read_next_command();
	}
	if (!committer)
		die("Expected committer but didn't get one");
	parse_data(&msg, 0, NULL);
	read_next_command();
	parse_from(b);
	merge_list = parse_merge(&merge_count);

	/* ensure the branch is active/loaded */
	if (!b->branch_tree.tree || !max_active_branches) {
		unload_one_branch();
		load_branch(b);
	}

	prev_fanout = convert_num_notes_to_fanout(b->num_notes);

	/* file_change* */
	while (command_buf.len > 0) {
		if (!prefixcmp(command_buf.buf, "M "))
			file_change_m(b);
		else if (!prefixcmp(command_buf.buf, "D "))
			file_change_d(b);
		else if (!prefixcmp(command_buf.buf, "R "))
			file_change_cr(b, 1);
		else if (!prefixcmp(command_buf.buf, "C "))
			file_change_cr(b, 0);
		else if (!prefixcmp(command_buf.buf, "N "))
			note_change_n(b, prev_fanout);
		else if (!strcmp("deleteall", command_buf.buf))
