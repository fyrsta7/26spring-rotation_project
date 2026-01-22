
		r = s_update_ref(msg, ref, 0);
		format_display(display, r ? '!' : '*', what,
			       r ? _("unable to update local ref") : NULL,
			       remote, pretty_ref, summary_width);
		return r;
	}

	if (fetch_show_forced_updates) {
		uint64_t t_before = getnanotime();
		fast_forward = in_merge_bases(current, updated);
		forced_updates_ms += (getnanotime() - t_before) / 1000000;
	} else {
		fast_forward = 1;
	}

	if (fast_forward) {
		struct strbuf quickref = STRBUF_INIT;
		int r;

		strbuf_add_unique_abbrev(&quickref, &current->object.oid, DEFAULT_ABBREV);
		strbuf_addstr(&quickref, "..");
		strbuf_add_unique_abbrev(&quickref, &ref->new_oid, DEFAULT_ABBREV);
		r = s_update_ref("fast-forward", ref, 1);
		format_display(display, r ? '!' : ' ', quickref.buf,
			       r ? _("unable to update local ref") : NULL,
			       remote, pretty_ref, summary_width);
		strbuf_release(&quickref);
		return r;
	} else if (force || ref->force) {
		struct strbuf quickref = STRBUF_INIT;
		int r;
		strbuf_add_unique_abbrev(&quickref, &current->object.oid, DEFAULT_ABBREV);
		strbuf_addstr(&quickref, "...");
		strbuf_add_unique_abbrev(&quickref, &ref->new_oid, DEFAULT_ABBREV);
		r = s_update_ref("forced-update", ref, 1);
		format_display(display, r ? '!' : '+', quickref.buf,
			       r ? _("unable to update local ref") : _("forced update"),
			       remote, pretty_ref, summary_width);
		strbuf_release(&quickref);
		return r;
	} else {
		format_display(display, '!', _("[rejected]"), _("non-fast-forward"),
			       remote, pretty_ref, summary_width);
		return 1;
	}
}

static int iterate_ref_map(void *cb_data, struct object_id *oid)
{
	struct ref **rm = cb_data;
	struct ref *ref = *rm;

	while (ref && ref->status == REF_STATUS_REJECT_SHALLOW)
		ref = ref->next;
	if (!ref)
		return -1; /* end of the list */
	*rm = ref->next;
	oidcpy(oid, &ref->old_oid);
	return 0;
}

static const char warn_show_forced_updates[] =
N_("Fetch normally indicates which branches had a forced update,\n"
   "but that check has been disabled. To re-enable, use '--show-forced-updates'\n"
   "flag or run 'git config fetch.showForcedUpdates true'.");
static const char warn_time_show_forced_updates[] =
N_("It took %.2f seconds to check forced updates. You can use\n"
   "'--no-show-forced-updates' or run 'git config fetch.showForcedUpdates false'\n"
   " to avoid this check.\n");

static int store_updated_refs(const char *raw_url, const char *remote_name,
			      int connectivity_checked, struct ref *ref_map)
{
	FILE *fp;
	struct commit *commit;
	int url_len, i, rc = 0;
	struct strbuf note = STRBUF_INIT;
	const char *what, *kind;
	struct ref *rm;
	char *url;
	const char *filename = (!write_fetch_head
				? "/dev/null"
				: git_path_fetch_head(the_repository));
	int want_status;
	int summary_width = transport_summary_width(ref_map);

	fp = fopen(filename, "a");
	if (!fp)
		return error_errno(_("cannot open %s"), filename);

	if (raw_url)
		url = transport_anonymize_url(raw_url);
	else
		url = xstrdup("foreign");

	if (!connectivity_checked) {
		struct check_connected_options opt = CHECK_CONNECTED_INIT;

		rm = ref_map;
		if (check_connected(iterate_ref_map, &rm, &opt)) {
			rc = error(_("%s did not send all necessary objects\n"), url);
			goto abort;
		}
	}

	prepare_format_display(ref_map);

	/*
	 * We do a pass for each fetch_head_status type in their enum order, so
	 * merged entries are written before not-for-merge. That lets readers
	 * use FETCH_HEAD as a refname to refer to the ref to be merged.
	 */
	for (want_status = FETCH_HEAD_MERGE;
	     want_status <= FETCH_HEAD_IGNORE;
	     want_status++) {
		for (rm = ref_map; rm; rm = rm->next) {
			struct ref *ref = NULL;
			const char *merge_status_marker = "";

			if (rm->status == REF_STATUS_REJECT_SHALLOW) {
				if (want_status == FETCH_HEAD_MERGE)
					warning(_("reject %s because shallow roots are not allowed to be updated"),
						rm->peer_ref ? rm->peer_ref->name : rm->name);
				continue;
			}

			commit = lookup_commit_reference_gently(the_repository,
								&rm->old_oid,
								1);
			if (!commit)
				rm->fetch_head_status = FETCH_HEAD_NOT_FOR_MERGE;

			if (rm->fetch_head_status != want_status)
				continue;

			if (rm->peer_ref) {
				ref = alloc_ref(rm->peer_ref->name);
				oidcpy(&ref->old_oid, &rm->peer_ref->old_oid);
				oidcpy(&ref->new_oid, &rm->old_oid);
				ref->force = rm->peer_ref->force;
			}

			if (recurse_submodules != RECURSE_SUBMODULES_OFF)
				check_for_new_submodule_commits(&rm->old_oid);

			if (!strcmp(rm->name, "HEAD")) {
				kind = "";
				what = "";
			}
			else if (skip_prefix(rm->name, "refs/heads/", &what))
				kind = "branch";
			else if (skip_prefix(rm->name, "refs/tags/", &what))
				kind = "tag";
			else if (skip_prefix(rm->name, "refs/remotes/", &what))
				kind = "remote-tracking branch";
			else {
				kind = "";
				what = rm->name;
			}

			url_len = strlen(url);
			for (i = url_len - 1; url[i] == '/' && 0 <= i; i--)
				;
