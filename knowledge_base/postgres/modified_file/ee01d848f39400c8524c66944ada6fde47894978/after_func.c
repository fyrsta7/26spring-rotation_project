			}
			result += rightmost_one_pos[w & 255];
		}
	}
	if (result < 0)
		elog(ERROR, "bitmapset is empty");
	return result;
}

/*
 * bms_num_members - count members of set
 */
int
bms_num_members(const Bitmapset *a)
{
	int			result = 0;
	int			nwords;
	int			wordnum;

	if (a == NULL)
		return 0;
	nwords = a->nwords;
	for (wordnum = 0; wordnum < nwords; wordnum++)
	{
		bitmapword	w = a->words[wordnum];

		/* we assume here that bitmapword is an unsigned type */
		while (w != 0)
		{
