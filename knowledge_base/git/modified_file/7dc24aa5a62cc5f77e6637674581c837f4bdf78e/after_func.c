		}
		while (cp < ep && *cp == sep)
			cp++;
		last = cp;
	}
}

static void read_info_alternates(const char * relative_base, int depth)
{
	char *map;
	size_t mapsz;
	struct stat st;
	char path[PATH_MAX];
	int fd;

