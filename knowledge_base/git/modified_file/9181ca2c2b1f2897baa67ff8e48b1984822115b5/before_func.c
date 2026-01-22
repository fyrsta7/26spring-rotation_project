	for ( ; list; list = list->next) {
		struct commit *commit = list->item;

		if (commit->object.flags & UNINTERESTING) {
			mark_tree_uninteresting(commit->tree);
			continue;
		}
		mark_edge_parents_uninteresting(commit);
	}
}

int main(int argc, const char **argv)
{
	struct commit_list *list;
	int i;

	argc = setup_revisions(argc, argv, &revs, NULL);

	for (i = 1 ; i < argc; i++) {
		const char *arg = argv[i];

		/* accept -<digit>, like traditilnal "head" */
		if ((*arg == '-') && isdigit(arg[1])) {
			revs.max_count = atoi(arg + 1);
			continue;
		}
		if (!strcmp(arg, "-n")) {
			if (++i >= argc)
				die("-n requires an argument");
			revs.max_count = atoi(argv[i]);
			continue;
		}
		if (!strncmp(arg,"-n",2)) {
			revs.max_count = atoi(arg + 2);
			continue;
		}
		if (!strcmp(arg, "--header")) {
			verbose_header = 1;
			continue;
		}
		if (!strcmp(arg, "--no-abbrev")) {
			abbrev = 0;
			continue;
		}
		if (!strncmp(arg, "--abbrev=", 9)) {
			abbrev = strtoul(arg + 9, NULL, 10);
			if (abbrev && abbrev < MINIMUM_ABBREV)
				abbrev = MINIMUM_ABBREV;
			else if (40 < abbrev)
				abbrev = 40;
			continue;
		}
		if (!strncmp(arg, "--pretty", 8)) {
			commit_format = get_commit_format(arg+8);
			verbose_header = 1;
			hdr_termination = '\n';
			if (commit_format == CMIT_FMT_ONELINE)
				commit_prefix = "";
			else
				commit_prefix = "commit ";
			continue;
		}
		if (!strcmp(arg, "--parents")) {
			show_parents = 1;
			continue;
		}
		if (!strcmp(arg, "--timestamp")) {
			show_timestamp = 1;
			continue;
		}
		if (!strcmp(arg, "--bisect")) {
			bisect_list = 1;
			continue;
		}
		usage(rev_list_usage);

	}

	list = revs.commits;

	if (!list &&
	    (!(revs.tag_objects||revs.tree_objects||revs.blob_objects) && !revs.pending_objects))
		usage(rev_list_usage);

	prepare_revision_walk(&revs);
	if (revs.tree_objects)
		mark_edges_uninteresting(revs.commits);
