			*bms = '\0';
		else
			break;

		++bms;
		++i;
	}
}

/*
 * Get the real path to a bookmark
 *
 * NULL is returned in case of no match, path resolution failure etc.
 * buf would be modified, so check return value before access
 */
static char *
get_bm_loc(char *key, char *buf)
{
	int r;

	if (!key || !key[0])
		return NULL;
