	index_file = prepare_index(argc, argv, prefix, 1);
	commitable = run_status(stdout, index_file, prefix, 0, s);
	rollback_index_files();

	return commitable ? 0 : 1;
}

static int parse_status_slot(const char *var, int offset)
{
	if (!strcasecmp(var+offset, "header"))
		return WT_STATUS_HEADER;
	if (!strcasecmp(var+offset, "updated")
		|| !strcasecmp(var+offset, "added"))
		return WT_STATUS_UPDATED;
	if (!strcasecmp(var+offset, "changed"))
		return WT_STATUS_CHANGED;
	if (!strcasecmp(var+offset, "untracked"))
		return WT_STATUS_UNTRACKED;
	if (!strcasecmp(var+offset, "nobranch"))
		return WT_STATUS_NOBRANCH;
	if (!strcasecmp(var+offset, "unmerged"))
		return WT_STATUS_UNMERGED;
	return -1;
}

static int git_status_config(const char *k, const char *v, void *cb)
{
	struct wt_status *s = cb;

	if (!strcmp(k, "status.submodulesummary")) {
		int is_bool;
		s->submodule_summary = git_config_bool_or_int(k, v, &is_bool);
		if (is_bool && s->submodule_summary)
			s->submodule_summary = -1;
		return 0;
	}
	if (!strcmp(k, "status.color") || !strcmp(k, "color.status")) {
		s->use_color = git_config_colorbool(k, v, -1);
		return 0;
	}
	if (!prefixcmp(k, "status.color.") || !prefixcmp(k, "color.status.")) {
		int slot = parse_status_slot(k, 13);
		if (slot < 0)
			return 0;
		if (!v)
			return config_error_nonbool(k);
		color_parse(v, k, s->color_palette[slot]);
		return 0;
	}
	if (!strcmp(k, "status.relativepaths")) {
		s->relative_paths = git_config_bool(k, v);
		return 0;
	}
	if (!strcmp(k, "status.showuntrackedfiles")) {
		if (!v)
			return config_error_nonbool(k);
		else if (!strcmp(v, "no"))
			s->show_untracked_files = SHOW_NO_UNTRACKED_FILES;
		else if (!strcmp(v, "normal"))
			s->show_untracked_files = SHOW_NORMAL_UNTRACKED_FILES;
		else if (!strcmp(v, "all"))
