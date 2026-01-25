void sha1_file_name(struct strbuf *buf, const unsigned char *sha1)
{
	strbuf_addstr(buf, get_object_directory());
	strbuf_addch(buf, '/');
	fill_sha1_path(buf, sha1);
}