static void feed_object(const struct object_id *oid, FILE *fh, int negative)
{
	if (negative &&
	    !has_object_file_with_flags(oid, OBJECT_INFO_SKIP_FETCH_OBJECT))
		return;

	if (negative)
		putc('^', fh);
	fputs(oid_to_hex(oid), fh);
	putc('\n', fh);
}
