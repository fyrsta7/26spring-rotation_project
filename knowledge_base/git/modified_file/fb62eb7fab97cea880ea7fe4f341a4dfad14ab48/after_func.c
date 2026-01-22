};

static int match_one_pattern(struct grep_opt *opt, struct grep_pat *p, char *bol, char *eol, enum grep_context ctx)
{
	int hit = 0;
	int saved_ch = 0;
	regmatch_t pmatch[10];

	if ((p->token != GREP_PATTERN) &&
	    ((p->token == GREP_PATTERN_HEAD) != (ctx == GREP_CONTEXT_HEAD)))
		return 0;

	if (p->token == GREP_PATTERN_HEAD) {
		const char *field;
		size_t len;
		assert(p->field < ARRAY_SIZE(header_field));
		field = header_field[p->field].field;
		len = header_field[p->field].len;
		if (strncmp(bol, field, len))
			return 0;
		bol += len;
		saved_ch = strip_timestamp(bol, &eol);
	}

 again:
	if (!opt->fixed) {
		regex_t *exp = &p->regexp;
		hit = !regexec(exp, bol, ARRAY_SIZE(pmatch),
			       pmatch, 0);
	}
	else {
		hit = !fixmatch(p->pattern, bol, pmatch);
	}

	if (hit && opt->word_regexp) {
		if ((pmatch[0].rm_so < 0) ||
		    (eol - bol) <= pmatch[0].rm_so ||
		    (pmatch[0].rm_eo < 0) ||
		    (eol - bol) < pmatch[0].rm_eo)
			die("regexp returned nonsense");

		/* Match beginning must be either beginning of the
		 * line, or at word boundary (i.e. the last char must
		 * not be a word char).  Similarly, match end must be
		 * either end of the line, or at word boundary
		 * (i.e. the next char must not be a word char).
		 */
		if ( ((pmatch[0].rm_so == 0) ||
		      !word_char(bol[pmatch[0].rm_so-1])) &&
		     ((pmatch[0].rm_eo == (eol-bol)) ||
		      !word_char(bol[pmatch[0].rm_eo])) )
			;
		else
			hit = 0;

		if (!hit && pmatch[0].rm_so + bol + 1 < eol) {
			/* There could be more than one match on the
			 * line, and the first match might not be
			 * strict word match.  But later ones could be!
			 * Forward to the next possible start, i.e. the
			 * next position following a non-word char.
