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
	char *url;
	struct strbuf buf = STRBUF_INIT;

	if (has_pack_index(sha1)) {
		ret = 0;
		goto cleanup;
	}

	if (http_is_verbose)
		fprintf(stderr, "Getting index for pack %s\n", hex);

	end_url_with_slash(&buf, base_url);
	strbuf_addf(&buf, "objects/pack/pack-%s.idx", hex);
	url = strbuf_detach(&buf, NULL);

	filename = sha1_pack_index_name(sha1);
