		return -1;

	/* if we are deltified, write out base object first. */
	if (e->delta && !write_one(f, e->delta, offset))
		return 0;

	e->idx.offset = *offset;
	size = write_object(f, e, *offset);
	if (!size) {
