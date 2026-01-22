	struct string_list *path_list = opt->output_priv;

	if (len == 1 && *(const char *)data == '\0')
		return;
	string_list_append(path_list, xstrndup(data, len));
}

static void run_pager(struct grep_opt *opt, const char *prefix)
{
	struct string_list *path_list = opt->output_priv;
	struct child_process child = CHILD_PROCESS_INIT;
	int i, status;

	for (i = 0; i < path_list->nr; i++)
		argv_array_push(&child.args, path_list->items[i].string);
	child.dir = prefix;
	child.use_shell = 1;

	status = run_command(&child);
	if (status)
		exit(status);
}

static int grep_cache(struct grep_opt *opt,
		      const struct pathspec *pathspec, int cached);
static int grep_tree(struct grep_opt *opt, const struct pathspec *pathspec,
		     struct tree_desc *tree, struct strbuf *base, int tn_len,
		     int check_attr);

static int grep_submodule(struct grep_opt *opt,
			  const struct pathspec *pathspec,
			  const struct object_id *oid,
			  const char *filename, const char *path, int cached)
{
	struct repository subrepo;
	struct repository *superproject = opt->repo;
	const struct submodule *sub;
	struct grep_opt subopt;
	int hit;

	sub = submodule_from_path(superproject, &null_oid, path);

	if (!is_submodule_active(superproject, path))
		return 0;

	if (repo_submodule_init(&subrepo, superproject, sub))
		return 0;

	/*
	 * NEEDSWORK: repo_read_gitmodules() might call
	 * add_to_alternates_memory() via config_from_gitmodules(). This
	 * operation causes a race condition with concurrent object readings
	 * performed by the worker threads. That's why we need obj_read_lock()
	 * here. It should be removed once it's no longer necessary to add the
	 * subrepo's odbs to the in-memory alternates list.
	 */
	obj_read_lock();
	repo_read_gitmodules(&subrepo, 0);

	/*
	 * NEEDSWORK: This adds the submodule's object directory to the list of
