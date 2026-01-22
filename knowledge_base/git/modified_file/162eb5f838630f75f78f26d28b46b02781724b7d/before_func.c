			ret = get_sha1_hex(buffer.buf, ref->old_sha1);
		else if (!prefixcmp(buffer.buf, "ref: ")) {
			ref->symref = xstrdup(buffer.buf + 5);
			ret = 0;
		}
	}

	strbuf_release(&buffer);
	free(url);
	return ret;
}

/* Helpers for fetching packs */
static int fetch_pack_index(unsigned char *sha1, const char *base_url)
{
	int ret = 0;
	char *hex = xstrdup(sha1_to_hex(sha1));
	char *filename;
	char *url = NULL;
	struct strbuf buf = STRBUF_INIT;

	if (has_pack_index(sha1)) {
		ret = 0;
		goto cleanup;
	}

	if (http_is_verbose)
		fprintf(stderr, "Getting index for pack %s\n", hex);

