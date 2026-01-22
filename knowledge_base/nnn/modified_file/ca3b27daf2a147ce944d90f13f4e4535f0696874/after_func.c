	if (c == -1) {
		++idle;

		/* Do not check for directory changes in du
		 * mode. A redraw forces du calculation.
		 * Check for changes every odd second.
		 */
#ifdef LINUX_INOTIFY
		if (!cfg.blkorder && inotify_wd >= 0 && idle & 1 && read(inotify_fd, inotify_buf, EVENT_BUF_LEN) > 0)
#elif defined(BSD_KQUEUE)
		if (!cfg.blkorder && event_fd >= 0 && idle & 1
		    && kevent(kq, events_to_monitor, NUM_EVENT_SLOTS, event_data, NUM_EVENT_FDS, &gtimeout) > 0)
#endif
				c = CONTROL('L');
	} else
		idle = 0;

	for (i = 0; i < len; ++i)
		if (c == bindings[i].sym) {
			*run = bindings[i].run;
			*env = bindings[i].env;
			return bindings[i].act;
		}

	return 0;
}

/*
 * Move non-matching entries to the end
 */
static int
fill(struct entry **dents, int (*filter)(regex_t *, char *), regex_t *re)
{
	static int count;
	static struct entry _dent, *pdent1, *pdent2;

	for (count = 0; count < ndents; ++count) {
		if (filter(re, (*dents)[count].name) == 0) {
			if (count != --ndents) {
				pdent1 = &(*dents)[count];
				pdent2 = &(*dents)[ndents];

				*(&_dent) = *pdent1;
				*pdent1 = *pdent2;
				*pdent2 = *(&_dent);
				--count;
			}

			continue;
		}
	}

	return ndents;
}

static int
matches(char *fltr)
{
	static regex_t re;

	/* Search filter */
	if (setfilter(&re, fltr) != 0)
		return -1;

	ndents = fill(&dents, visible, &re);
	regfree(&re);
	if (ndents == 0)
		return 0;

	qsort(dents, ndents, sizeof(*dents), entrycmp);

	return 0;
}

static int
filterentries(char *path)
{
	static char ln[REGEX_MAX] __attribute__ ((aligned));
	static wchar_t wln[REGEX_MAX] __attribute__ ((aligned));
	static wint_t ch[2] = {0};
	int r, total = ndents, oldcur = cur, len = 1;
	char *pln = ln + 1;

	ln[0] = wln[0] = FILTER;
	ln[1] = wln[1] = '\0';
	cur = 0;

	cleartimeout();
	echo();
	curs_set(TRUE);
	printprompt(ln);

	while ((r = get_wch(ch)) != ERR) {
		if (*ch == 127 /* handle DEL */ || *ch == KEY_DC || *ch == KEY_BACKSPACE || *ch == '\b') {
			if (len == 1) {
				cur = oldcur;
				*ch = CONTROL('L');
				goto end;
			}

			wln[--len] = '\0';
			if (len == 1)
				cur = oldcur;

			wcstombs(ln, wln, REGEX_MAX);
			ndents = total;
			if (matches(pln) == -1)
				continue;
			redraw(path);
			printprompt(ln);
			continue;
		}

		if (r == OK) {
			/* Handle all control chars in main loop */
			if (keyname(*ch)[0] == '^') {
				if (len == 1)
					cur = oldcur;
				goto end;
			}

			switch (*ch) {
			case '\r':  // with nonl(), this is ENTER key value
				if (len == 1) {
					cur = oldcur;
					goto end;
				}
