		break;
	}
	return -1;
}

static struct cache_entry *find_cache_entry(struct traverse_info *info,
					    const struct name_entry *p)
{
	int pos = find_cache_pos(info, p->path, p->pathlen);
	struct unpack_trees_options *o = info->data;

	if (0 <= pos)
		return o->src_index->cache[pos];
	else
		return NULL;
}

static void debug_path(struct traverse_info *info)
{
	if (info->prev) {
		debug_path(info->prev);
		if (*info->prev->name)
			putchar('/');
	}
	printf("%s", info->name);
}

static void debug_name_entry(int i, struct name_entry *n)
{
	printf("ent#%d %06o %s\n", i,
	       n->path ? n->mode : 0,
	       n->path ? n->path : "(missing)");
}

static void debug_unpack_callback(int n,
				  unsigned long mask,
				  unsigned long dirmask,
				  struct name_entry *names,
				  struct traverse_info *info)
{
	int i;
	printf("* unpack mask %lu, dirmask %lu, cnt %d ",
	       mask, dirmask, n);
	debug_path(info);
	putchar('\n');
	for (i = 0; i < n; i++)
		debug_name_entry(i, names + i);
}

/*
 * Note that traverse_by_cache_tree() duplicates some logic in this function
