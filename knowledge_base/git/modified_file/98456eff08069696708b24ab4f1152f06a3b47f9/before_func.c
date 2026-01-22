
static int ls_refs_config(const char *var, const char *value,
			  void *cb_data)
{
	struct ls_refs_data *data = cb_data;
	/*
	 * We only serve fetches over v2 for now, so respect only "uploadpack"
	 * config. This may need to eventually be expanded to "receive", but we
	 * don't yet know how that information will be passed to ls-refs.
	 */
	return parse_hide_refs_config(var, value, "uploadpack", &data->hidden_refs);
}

int ls_refs(struct repository *r, struct packet_reader *request)
{
	struct ls_refs_data data;

	memset(&data, 0, sizeof(data));
	strvec_init(&data.prefixes);
	strbuf_init(&data.buf, 0);
	strvec_init(&data.hidden_refs);

	git_config(ls_refs_config, &data);

	while (packet_reader_read(request) == PACKET_READ_NORMAL) {
		const char *arg = request->line;
		const char *out;

		if (!strcmp("peel", arg))
			data.peel = 1;
		else if (!strcmp("symrefs", arg))
			data.symrefs = 1;
		else if (skip_prefix(arg, "ref-prefix ", &out)) {
			if (data.prefixes.nr < TOO_MANY_PREFIXES)
				strvec_push(&data.prefixes, out);
		}
		else if (!strcmp("unborn", arg))
			data.unborn = !!unborn_config(r);
		else
			die(_("unexpected line: '%s'"), arg);
	}

	if (request->status != PACKET_READ_FLUSH)
		die(_("expected flush after ls-refs arguments"));

	/*
	 * If we saw too many prefixes, we must avoid using them at all; as
	 * soon as we have any prefix, they are meant to form a comprehensive
