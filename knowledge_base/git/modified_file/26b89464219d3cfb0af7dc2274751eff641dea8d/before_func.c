	}

	if (untracked->check_only != !!check_only)
		return 0;

	/*
	 * prep_exclude will be called eventually on this directory,
	 * but it's called much later in last_matching_pattern(). We
	 * need it now to determine the validity of the cache for this
	 * path. The next calls will be nearly no-op, the way
	 * prep_exclude() is designed.
	 */
	if (path->len && path->buf[path->len - 1] != '/') {
		strbuf_addch(path, '/');
		prep_exclude(dir, istate, path->buf, path->len);
		strbuf_setlen(path, path->len - 1);
	} else
		prep_exclude(dir, istate, path->buf, path->len);

	/* hopefully prep_exclude() haven't invalidated this entry... */
	return untracked->valid;
}

static int open_cached_dir(struct cached_dir *cdir,
			   struct dir_struct *dir,
			   struct untracked_cache_dir *untracked,
			   struct index_state *istate,
			   struct strbuf *path,
			   int check_only)
{
	const char *c_path;

	memset(cdir, 0, sizeof(*cdir));
	cdir->untracked = untracked;
	if (valid_cached_dir(dir, untracked, istate, path, check_only))
		return 0;
	c_path = path->len ? path->buf : ".";
	cdir->fdir = opendir(c_path);
	if (!cdir->fdir)
		warning_errno(_("could not open directory '%s'"), c_path);
	if (dir->untracked) {
		invalidate_directory(dir->untracked, untracked);
		dir->untracked->dir_opened++;
	}
	if (!cdir->fdir)
