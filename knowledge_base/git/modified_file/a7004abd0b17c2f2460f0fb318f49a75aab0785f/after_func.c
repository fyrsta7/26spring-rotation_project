	set_read_ref_cutoffs(cb, timestamp, tz, message);
	oidcpy(cb->oid, ooid);
	if (cb->at_time && is_null_oid(cb->oid))
		oidcpy(cb->oid, noid);
	/* We just want the first entry */
	return 1;
}

int read_ref_at(struct ref_store *refs, const char *refname,
		unsigned int flags, timestamp_t at_time, int cnt,
		struct object_id *oid, char **msg,
		timestamp_t *cutoff_time, int *cutoff_tz, int *cutoff_cnt)
{
	struct read_ref_at_cb cb;

	memset(&cb, 0, sizeof(cb));
	cb.refname = refname;
	cb.at_time = at_time;
	cb.cnt = cnt;
	cb.msg = msg;
	cb.cutoff_time = cutoff_time;
	cb.cutoff_tz = cutoff_tz;
	cb.cutoff_cnt = cutoff_cnt;
	cb.oid = oid;

	refs_for_each_reflog_ent_reverse(refs, refname, read_ref_at_ent, &cb);

	if (!cb.reccnt) {
		if (cnt == 0) {
			/*
			 * The caller asked for ref@{0}, and we had no entries.
			 * It's a bit subtle, but in practice all callers have
			 * prepped the "oid" field with the current value of
			 * the ref, which is the most reasonable fallback.
			 *
