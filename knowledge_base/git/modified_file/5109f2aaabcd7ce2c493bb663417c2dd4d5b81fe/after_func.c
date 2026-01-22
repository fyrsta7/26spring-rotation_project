}

int for_each_abbrev(const char *prefix, each_abbrev_fn fn, void *cb_data)
{
	char hex_pfx[40];
	unsigned char bin_pfx[20];
	struct disambiguate_state ds;
	int len = strlen(prefix);

	if (len < MINIMUM_ABBREV || len > 40)
		return -1;
	if (prepare_prefixes(prefix, len, bin_pfx, hex_pfx) < 0)
		return -1;

	prepare_alt_odb();

	memset(&ds, 0, sizeof(ds));
	ds.always_call_fn = 1;
	ds.cb_data = cb_data;
	ds.fn = fn;

	find_short_object_filename(len, hex_pfx, &ds);
