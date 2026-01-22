	map = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
	close(fd);
	if (map == MAP_FAILED)
		return;

	link_alt_odb_entries(map, map + st.st_size, '\n',
			     get_object_directory());
	munmap(map, st.st_size);
}

static char *find_sha1_file(const unsigned char *sha1, struct stat *st)
{
	char *name = sha1_file_name(sha1);
	struct alternate_object_database *alt;

	if (!stat(name, st))
		return name;
	prepare_alt_odb();
	for (alt = alt_odb_list; alt; alt = alt->next) {
		name = alt->name;
		fill_sha1_path(name, sha1);
		if (!stat(alt->base, st))
			return alt->base;
	}
	return NULL;
}

#define PACK_MAX_SZ (1<<26)
static int pack_used_ctr;
static unsigned long pack_mapped;
struct packed_git *packed_git;

static int check_packed_git_idx(const char *path, unsigned long *idx_size_,
				void **idx_map_)
{
	void *idx_map;
	unsigned int *index;
	unsigned long idx_size;
	int nr, i;
	int fd = open(path, O_RDONLY);
	struct stat st;
	if (fd < 0)
