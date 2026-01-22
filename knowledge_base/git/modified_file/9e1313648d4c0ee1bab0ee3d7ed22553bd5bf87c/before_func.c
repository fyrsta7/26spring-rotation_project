
static void prepare_revs(struct replay_opts *opts)
{
	if (opts->action != REVERT)
		opts->revs->reverse ^= 1;

	if (prepare_revision_walk(opts->revs))
		die(_("revision walk setup failed"));

	if (!opts->revs->commits)
		die(_("empty commit set passed"));
}

static void read_and_refresh_cache(struct replay_opts *opts)
{
	static struct lock_file index_lock;
	int index_fd = hold_locked_index(&index_lock, 0);
