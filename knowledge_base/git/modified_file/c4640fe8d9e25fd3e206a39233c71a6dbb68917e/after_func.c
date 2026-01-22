	if (encoding_header_len < need_len) {
		buf = xrealloc(buf, buflen + (need_len - encoding_header_len));
		encoding_header = buf + encoding_header_pos;
		end_of_encoding_header = encoding_header + encoding_header_len;
	}
	memmove(end_of_encoding_header + (need_len - encoding_header_len),
		end_of_encoding_header,
		buflen - (encoding_header_pos + encoding_header_len));
	memcpy(encoding_header + 9, encoding, strlen(encoding));
	encoding_header[9 + new_len] = '\n';
	return buf;
}

static char *logmsg_reencode(const struct commit *commit,
			     const char *output_encoding)
{
	static const char *utf8 = "utf-8";
	const char *use_encoding;
	char *encoding;
	char *out;

	if (!*output_encoding)
		return NULL;
	encoding = get_header(commit, "encoding");
	use_encoding = encoding ? encoding : utf8;
	if (!strcmp(use_encoding, output_encoding))
