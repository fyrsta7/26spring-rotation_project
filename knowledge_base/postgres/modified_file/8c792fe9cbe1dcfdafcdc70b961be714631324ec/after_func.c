
static int MatchText(unsigned char *t, int tlen,
		  unsigned char *p, int plen);
static int MatchTextIC(unsigned char *t, int tlen,
			unsigned char *p, int plen);
static int MatchBytea(unsigned char *t, int tlen,
		   unsigned char *p, int plen);
static text *do_like_escape(text *, text *);

static int MBMatchText(unsigned char *t, int tlen,
			unsigned char *p, int plen);
static int MBMatchTextIC(unsigned char *t, int tlen,
			  unsigned char *p, int plen);
static text *MB_do_like_escape(text *, text *);

/*--------------------
 * Support routine for MatchText. Compares given multibyte streams
 * as wide characters. If they match, returns 1 otherwise returns 0.
 *--------------------
 */
static int
