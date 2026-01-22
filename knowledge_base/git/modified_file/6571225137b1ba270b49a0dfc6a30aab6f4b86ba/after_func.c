
	if (buffer.len == 0)
		return;

	/* If it's a symref, set the refname; otherwise try for a sha1 */
	if (!prefixcmp((char *)buffer.buf, "ref: ")) {
		*symref = xmemdupz((char *)buffer.buf + 5, buffer.len - 6);
