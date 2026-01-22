			*((void**)e) = e + 1;
			e++;
		}
		*((void**)e) = NULL;
	}

	e = avail_tree_entry;
	avail_tree_entry = *((void**)e);
	return e;
}

static void release_tree_entry(struct tree_entry *e)
{
	if (e->tree)
		release_tree_content_recursive(e->tree);
	*((void**)e) = avail_tree_entry;
	avail_tree_entry = e;
}

static void yread(int fd, void *buffer, size_t length)
{
	ssize_t ret = 0;
	while (ret < length) {
		ssize_t size = xread(fd, (char *) buffer + ret, length - ret);
		if (!size)
			die("Read from descriptor %i: end of stream", fd);
		if (size < 0)
			die("Read from descriptor %i: %s", fd, strerror(errno));
		ret += size;
	}
}

static void start_packfile()
{
	struct packed_git *p;
	struct pack_header hdr;

	idx_name = xmalloc(strlen(base_name) + 11);
	p = xcalloc(1, sizeof(*p) + strlen(base_name) + 13);
	sprintf(p->pack_name, "%s%5.5i.pack", base_name, pack_id + 1);
	sprintf(idx_name, "%s%5.5i.idx", base_name, pack_id + 1);

