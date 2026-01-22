{
	xrecord_t **recs;
	int size = 0;

	recs = (use_orig ? xe->xdf1.recs : xe->xdf2.recs) + i;

	if (count < 1)
		return 0;

	for (i = 0; i < count; size += recs[i++]->size)
		if (dest)
			memcpy(dest + size, recs[i]->ptr, recs[i]->size);
	if (add_nl) {
		i = recs[count - 1]->size;
		if (i == 0 || recs[count - 1]->ptr[i - 1] != '\n') {
			if (dest)
				dest[size] = '\n';
			size++;
		}
	}
	return size;
}

static int xdl_recs_copy(xdfenv_t *xe, int i, int count, int add_nl, char *dest)
{
	return xdl_recs_copy_0(0, xe, i, count, add_nl, dest);
}

static int xdl_orig_copy(xdfenv_t *xe, int i, int count, int add_nl, char *dest)
{
	return xdl_recs_copy_0(1, xe, i, count, add_nl, dest);
}

static int fill_conflict_hunk(xdfenv_t *xe1, const char *name1,
			      xdfenv_t *xe2, const char *name2,
			      const char *name3,
			      int size, int i, int style,
			      xdmerge_t *m, char *dest, int marker_size)
{
	int marker1_size = (name1 ? strlen(name1) + 1 : 0);
	int marker2_size = (name2 ? strlen(name2) + 1 : 0);
	int marker3_size = (name3 ? strlen(name3) + 1 : 0);

	if (marker_size <= 0)
		marker_size = DEFAULT_CONFLICT_MARKER_SIZE;

	/* Before conflicting part */
	size += xdl_recs_copy(xe1, i, m->i1 - i, 0,
			      dest ? dest + size : NULL);

	if (!dest) {
		size += marker_size + 1 + marker1_size;
	} else {
		memset(dest + size, '<', marker_size);
		size += marker_size;
		if (marker1_size) {
			dest[size] = ' ';
			memcpy(dest + size + 1, name1, marker1_size - 1);
			size += marker1_size;
		}
		dest[size++] = '\n';
	}

	/* Postimage from side #1 */
	size += xdl_recs_copy(xe1, m->i1, m->chg1, 1,
			      dest ? dest + size : NULL);

	if (style == XDL_MERGE_DIFF3) {
		/* Shared preimage */
		if (!dest) {
			size += marker_size + 1 + marker3_size;
		} else {
			memset(dest + size, '|', marker_size);
			size += marker_size;
			if (marker3_size) {
				dest[size] = ' ';
				memcpy(dest + size + 1, name3, marker3_size - 1);
