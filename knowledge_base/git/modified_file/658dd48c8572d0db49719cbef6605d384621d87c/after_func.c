		oldmode = ce->ce_mode;
		newmode = ce_mode_from_stat(ce, st.st_mode);
		diff_change(&revs->diffopt, oldmode, newmode,
			    ce->sha1, (changed ? null_sha1 : ce->sha1),
			    ce->name);

	}
	diffcore_std(&revs->diffopt);
	diff_flush(&revs->diffopt);
	return 0;
}

/*
 * diff-index
 */

struct oneway_unpack_data {
	struct rev_info *revs;
	char symcache[PATH_MAX];
};

/* A file entry went away or appeared */
static void diff_index_show_file(struct rev_info *revs,
				 const char *prefix,
				 struct cache_entry *ce,
				 const unsigned char *sha1, unsigned int mode)
{
	diff_addremove(&revs->diffopt, prefix[0], mode,
		       sha1, ce->name);
}

static int get_stat_data(struct cache_entry *ce,
			 const unsigned char **sha1p,
			 unsigned int *modep,
