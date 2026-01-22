	 */
	trace_printf_key(&trace_fsmonitor, "fsmonitor_refresh_callback '%s'", name);
	untracked_cache_invalidate_path(istate, name, 0);
}

void refresh_fsmonitor(struct index_state *istate)
{
	static int has_run_once = 0;
	struct strbuf query_result = STRBUF_INIT;
	int query_success = 0;
	size_t bol; /* beginning of line */
	uint64_t last_update;
	char *buf;
	int i;

	if (!core_fsmonitor || has_run_once)
		return;
	has_run_once = 1;

	trace_printf_key(&trace_fsmonitor, "refresh fsmonitor");
	/*
	 * This could be racy so save the date/time now and query_fsmonitor
	 * should be inclusive to ensure we don't miss potential changes.
	 */
	last_update = getnanotime();

	/*
	 * If we have a last update time, call query_fsmonitor for the set of
	 * changes since that time, else assume everything is possibly dirty
	 * and check it all.
	 */
	if (istate->fsmonitor_last_update) {
		query_success = !query_fsmonitor(HOOK_INTERFACE_VERSION,
			istate->fsmonitor_last_update, &query_result);
		trace_performance_since(last_update, "fsmonitor process '%s'", core_fsmonitor);
		trace_printf_key(&trace_fsmonitor, "fsmonitor process '%s' returned %s",
			core_fsmonitor, query_success ? "success" : "failure");
	}

	/* a fsmonitor process can return '/' to indicate all entries are invalid */
	if (query_success && query_result.buf[0] != '/') {
		/* Mark all entries returned by the monitor as dirty */
		buf = query_result.buf;
		bol = 0;
		for (i = 0; i < query_result.len; i++) {
			if (buf[i] != '\0')
				continue;
			fsmonitor_refresh_callback(istate, buf + bol);
			bol = i + 1;
		}
		if (bol < query_result.len)
			fsmonitor_refresh_callback(istate, buf + bol);
	} else {
		/* Mark all entries invalid */
		for (i = 0; i < istate->cache_nr; i++)
			istate->cache[i]->ce_flags &= ~CE_FSMONITOR_VALID;
