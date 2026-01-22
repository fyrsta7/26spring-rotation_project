
	if (opt->extended)
		return 0; /* punt for too complex stuff */
	if (opt->invert)
		return 0;
	for (p = opt->pattern_list; p; p = p->next) {
		if (p->token != GREP_PATTERN)
			return 0; /* punt for "header only" and stuff */
	}
	return 1;
}

static int look_ahead(struct grep_opt *opt,
		      unsigned long *left_p,
		      unsigned *lno_p,
		      char **bol_p)
{
	unsigned lno = *lno_p;
	char *bol = *bol_p;
	struct grep_pat *p;
	char *sp, *last_bol;
	regoff_t earliest = -1;

	for (p = opt->pattern_list; p; p = p->next) {
		int hit;
		regmatch_t m;

		if (p->fixed)
			hit = !fixmatch(p->pattern, bol, p->ignore_case, &m);
		else {
#ifdef REG_STARTEND
			m.rm_so = 0;
			m.rm_eo = *left_p;
			hit = !regexec(&p->regexp, bol, 1, &m, REG_STARTEND);
#else
			hit = !regexec(&p->regexp, bol, 1, &m, 0);
#endif
		}
		if (!hit || m.rm_so < 0 || m.rm_eo < 0)
			continue;
		if (earliest < 0 || m.rm_so < earliest)
			earliest = m.rm_so;
	}

	if (earliest < 0) {
		*bol_p = bol + *left_p;
		*left_p = 0;
		return 1;
	}
	for (sp = bol + earliest; bol < sp && sp[-1] != '\n'; sp--)
