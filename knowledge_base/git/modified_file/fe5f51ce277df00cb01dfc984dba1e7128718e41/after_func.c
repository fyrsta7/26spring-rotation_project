		}
		add_pending_object(object, "");
		return NULL;
	}
	die("%s is unknown object", name);
}

static void handle_one_commit(struct commit *com, struct commit_list **lst)
{
	if (!com || com->object.flags & SEEN)
		return;
	com->object.flags |= SEEN;
	commit_list_insert(com, lst);
}

/* for_each_ref() callback does not allow user data -- Yuck. */
static struct commit_list **global_lst;

static int include_one_commit(const char *path, const unsigned char *sha1)
{
	struct commit *com = get_commit_reference(path, 0);
	handle_one_commit(com, global_lst);
	return 0;
}

static void handle_all(struct commit_list **lst)
{
	global_lst = lst;
	for_each_ref(include_one_commit);
	global_lst = NULL;
}

int main(int argc, char **argv)
{
	struct commit_list *list = NULL;
	int i, limited = 0;

	setup_git_directory();
	for (i = 1 ; i < argc; i++) {
		int flags;
		char *arg = argv[i];
		char *dotdot;
		struct commit *commit;

		if (!strncmp(arg, "--max-count=", 12)) {
			max_count = atoi(arg + 12);
			continue;
		}
		if (!strncmp(arg, "--max-age=", 10)) {
			max_age = atoi(arg + 10);
			limited = 1;
			continue;
		}
		if (!strncmp(arg, "--min-age=", 10)) {
			min_age = atoi(arg + 10);
			limited = 1;
			continue;
		}
		if (!strcmp(arg, "--header")) {
			verbose_header = 1;
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
		if (!strncmp(arg, "--no-merges", 11)) {
			no_merges = 1;
			continue;
		}
		if (!strcmp(arg, "--parents")) {
			show_parents = 1;
			continue;
		}
		if (!strcmp(arg, "--bisect")) {
			bisect_list = 1;
			continue;
		}
		if (!strcmp(arg, "--all")) {
			handle_all(&list);
			continue;
		}
		if (!strcmp(arg, "--objects")) {
			tag_objects = 1;
			tree_objects = 1;
			blob_objects = 1;
			continue;
		}
		if (!strcmp(arg, "--unpacked")) {
			unpacked = 1;
			limited = 1;
			continue;
		}
		if (!strcmp(arg, "--merge-order")) {
		        merge_order = 1;
			continue;
		}
		if (!strcmp(arg, "--show-breaks")) {
			show_breaks = 1;
			continue;
		}
		if (!strcmp(arg, "--topo-order")) {
		        topo_order = 1;
		        limited = 1;
			continue;
		}

		if (show_breaks && !merge_order)
			usage(rev_list_usage);

		flags = 0;
		dotdot = strstr(arg, "..");
		if (dotdot) {
			char *next = dotdot + 2;
			struct commit *exclude = NULL;
			struct commit *include = NULL;
			*dotdot = 0;
			if (!*next)
				next = "HEAD";
			exclude = get_commit_reference(arg, UNINTERESTING);
			include = get_commit_reference(next, 0);
			if (exclude && include) {
				limited = 1;
				handle_one_commit(exclude, &list);
				handle_one_commit(include, &list);
				continue;
			}
			*dotdot = '.';
		}
		if (*arg == '^') {
			flags = UNINTERESTING;
			arg++;
			limited = 1;
