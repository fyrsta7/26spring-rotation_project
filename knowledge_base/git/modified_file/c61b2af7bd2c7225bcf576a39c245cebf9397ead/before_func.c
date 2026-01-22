 * stream, aka "verbose").  A message over band #3 is a signal that
 * the remote died unexpectedly.  A flush() concludes the stream.
 */

#define PREFIX "remote: "

#define ANSI_SUFFIX "\033[K"
#define DUMB_SUFFIX "        "

int recv_sideband(const char *me, int in_stream, int out)
{
	const char *term, *suffix;
	char buf[LARGE_PACKET_MAX + 1];
	struct strbuf outbuf = STRBUF_INIT;
	int retval = 0;

	term = getenv("TERM");
	if (isatty(2) && term && strcmp(term, "dumb"))
		suffix = ANSI_SUFFIX;
	else
		suffix = DUMB_SUFFIX;

	while (!retval) {
		const char *b, *brk;
		int band, len;
		len = packet_read(in_stream, NULL, NULL, buf, LARGE_PACKET_MAX, 0);
		if (len == 0)
			break;
		if (len < 1) {
			strbuf_addf(&outbuf,
				    "%s%s: protocol error: no band designator",
				    outbuf.len ? "\n" : "", me);
			retval = SIDEBAND_PROTOCOL_ERROR;
			break;
		}
		band = buf[0] & 0xff;
		buf[len] = '\0';
		len--;
		switch (band) {
		case 3:
			strbuf_addf(&outbuf, "%s%s%s", outbuf.len ? "\n" : "",
				    PREFIX, buf + 1);
			retval = SIDEBAND_REMOTE_ERROR;
			break;
		case 2:
			b = buf + 1;

			/*
			 * Append a suffix to each nonempty line to clear the
			 * end of the screen line.
			 *
			 * The output is accumulated in a buffer and
			 * each line is printed to stderr using
			 * write(2) to ensure inter-process atomicity.
			 */
			while ((brk = strpbrk(b, "\n\r"))) {
				int linelen = brk - b;

				if (!outbuf.len)
					strbuf_addf(&outbuf, "%s", PREFIX);
				if (linelen > 0) {
					strbuf_addf(&outbuf, "%.*s%s%c",
						    linelen, b, suffix, *brk);
				} else {
					strbuf_addf(&outbuf, "%c", *brk);
				}
				xwrite(2, outbuf.buf, outbuf.len);
				strbuf_reset(&outbuf);

				b = brk + 1;
			}

			if (*b)
				strbuf_addf(&outbuf, "%s%s",
					    outbuf.len ? "" : PREFIX, b);
			break;
		case 1:
			write_or_die(out, buf + 1, len);
