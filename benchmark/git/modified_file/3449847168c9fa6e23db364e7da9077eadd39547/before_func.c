void sha1_file_name(struct strbuf *buf, const unsigned char *sha1)
{
	strbuf_addf(buf, "%s/", get_object_directory());

	fill_sha1_path(buf, sha1);
}