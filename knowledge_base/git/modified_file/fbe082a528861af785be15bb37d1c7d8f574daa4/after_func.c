	diff_setup(reverse_diff);
}

static int call_diff_flush(void)
{
	if (detect_rename)
		diffcore_rename(detect_rename, diff_score_opt);
	if (pickaxe)
		diffcore_pickaxe(pickaxe);
	if (diff_queue_is_empty()) {
		diff_flush(DIFF_FORMAT_NO_OUTPUT, 0);
		return 0;
	}
	if (header) {
		if (diff_output_format == DIFF_FORMAT_MACHINE) {
			const char *ep, *cp;
			for (cp = header; *cp; cp = ep) {
				ep = strchr(cp, '\n');
				if (ep == 0) ep = cp + strlen(cp);
				printf("%.*s%c", ep-cp, cp, 0);
				if (*ep) ep++;
			}
		}
		else {
			printf("%s", header);
		}
		header = NULL;
	}
