		 * caller early.
		 */
		return;
	/* Yuck -- line ought to be "const char *"! */
	hold = line[len];
	line[len] = '\0';
	data->hit = !regexec(data->regexp, line + 1, 1, &regmatch, 0);
	line[len] = hold;
}

static int diff_grep(struct diff_filepair *p, struct diff_options *o,
		     regex_t *regexp, kwset_t kws)
{
	regmatch_t regmatch;
	struct userdiff_driver *textconv_one = NULL;
	struct userdiff_driver *textconv_two = NULL;
	mmfile_t mf1, mf2;
	int hit;

	if (!o->pickaxe[0])
		return 0;

	if (DIFF_OPT_TST(o, ALLOW_TEXTCONV)) {
		textconv_one = get_textconv(p->one);
		textconv_two = get_textconv(p->two);
	}

	if (textconv_one == textconv_two && diff_unmodified_pair(p))
		return 0;

	mf1.size = fill_textconv(textconv_one, p->one, &mf1.ptr);
	mf2.size = fill_textconv(textconv_two, p->two, &mf2.ptr);

	if (!DIFF_FILE_VALID(p->one)) {
		if (!DIFF_FILE_VALID(p->two))
			return 0; /* ignore unmerged */
		/* created "two" -- does it have what we are looking for? */
		hit = !regexec(regexp, mf2.ptr, 1, &regmatch, 0);
	} else if (!DIFF_FILE_VALID(p->two)) {
		/* removed "one" -- did it have what we are looking for? */
		hit = !regexec(regexp, mf1.ptr, 1, &regmatch, 0);
	} else {
		/*
		 * We have both sides; need to run textual diff and see if
		 * the pattern appears on added/deleted lines.
		 */
		struct diffgrep_cb ecbdata;
		xpparam_t xpp;
		xdemitconf_t xecfg;

		memset(&xpp, 0, sizeof(xpp));
		memset(&xecfg, 0, sizeof(xecfg));
		ecbdata.regexp = regexp;
		ecbdata.hit = 0;
		xecfg.ctxlen = o->context;
