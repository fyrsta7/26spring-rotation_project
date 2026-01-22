		default:
			slash = 0;
			continue;
		}
		break;
	}
	return slash;
}

static inline int upstream_mark(const char *string, int len)
{
	const char *suffix[] = { "@{upstream}", "@{u}" };
	int i;

	for (i = 0; i < ARRAY_SIZE(suffix); i++) {
		int suffix_len = strlen(suffix[i]);
		if (suffix_len <= len
		    && !memcmp(string, suffix[i], suffix_len))
			return suffix_len;
	}
	return 0;
}

static int get_sha1_1(const char *name, int len, unsigned char *sha1, unsigned lookup_flags);
static int interpret_nth_prior_checkout(const char *name, struct strbuf *buf);

static int get_sha1_basic(const char *str, int len, unsigned char *sha1)
{
	static const char *warn_msg = "refname '%.*s' is ambiguous.";
	static const char *object_name_msg = N_(
	"Git normally never creates a ref that ends with 40 hex characters\n"
	"because it will be ignored when you just specify 40-hex. These refs\n"
	"may be created by mistake. For example,\n"
	"\n"
	"  git checkout -b $br $(git rev-parse ...)\n"
	"\n"
	"where \"$br\" is somehow empty and a 40-hex ref is created. Please\n"
	"examine these refs and maybe delete them. Turn this message off by\n"
	"running \"git config advice.objectNameWarning false\"");
	unsigned char tmp_sha1[20];
	char *real_ref = NULL;
	int refs_found = 0;
	int at, reflog_len, nth_prior = 0;

	if (len == 40 && !get_sha1_hex(str, sha1)) {
		if (warn_ambiguous_refs && warn_on_object_refname_ambiguity) {
			refs_found = dwim_ref(str, len, tmp_sha1, &real_ref);
			if (refs_found > 0) {
				warning(warn_msg, len, str);
				if (advice_object_name_warning)
					fprintf(stderr, "%s\n", _(object_name_msg));
			}
			free(real_ref);
		}
		return 0;
	}

	/* basic@{time or number or -number} format to query ref-log */
	reflog_len = at = 0;
	if (len && str[len-1] == '}') {
		for (at = len-4; at >= 0; at--) {
			if (str[at] == '@' && str[at+1] == '{') {
				if (str[at+2] == '-') {
					if (at != 0)
						/* @{-N} not at start */
						return -1;
					nth_prior = 1;
					continue;
				}
				if (!upstream_mark(str + at, len - at)) {
					reflog_len = (len-1) - (at+2);
					len = at;
				}
				break;
			}
		}
	}

	/* Accept only unambiguous ref paths. */
	if (len && ambiguous_path(str, len))
		return -1;

	if (nth_prior) {
		struct strbuf buf = STRBUF_INIT;
		int detached;

		if (interpret_nth_prior_checkout(str, &buf) > 0) {
			detached = (buf.len == 40 && !get_sha1_hex(buf.buf, sha1));
			strbuf_release(&buf);
			if (detached)
				return 0;
		}
	}

	if (!len && reflog_len)
		/* allow "@{...}" to mean the current branch reflog */
		refs_found = dwim_ref("HEAD", 4, sha1, &real_ref);
	else if (reflog_len)
		refs_found = dwim_log(str, len, sha1, &real_ref);
	else
		refs_found = dwim_ref(str, len, sha1, &real_ref);

	if (!refs_found)
		return -1;

	if (warn_ambiguous_refs &&
	    (refs_found > 1 ||
	     !get_short_sha1(str, len, tmp_sha1, GET_SHA1_QUIETLY)))
		warning(warn_msg, len, str);

	if (reflog_len) {
		int nth, i;
		unsigned long at_time;
		unsigned long co_time;
		int co_tz, co_cnt;

		/* Is it asking for N-th entry, or approxidate? */
		for (i = nth = 0; 0 <= nth && i < reflog_len; i++) {
			char ch = str[at+2+i];
			if ('0' <= ch && ch <= '9')
				nth = nth * 10 + ch - '0';
			else
				nth = -1;
		}
		if (100000000 <= nth) {
			at_time = nth;
			nth = -1;
		} else if (0 <= nth)
			at_time = 0;
		else {
			int errors = 0;
			char *tmp = xstrndup(str + at + 2, reflog_len);
			at_time = approxidate_careful(tmp, &errors);
			free(tmp);
			if (errors)
				return -1;
		}
