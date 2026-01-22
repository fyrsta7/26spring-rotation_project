 * between '+' and '-' lines that have been detected to be a move.
 * The string contains the difference in leading white spaces, before the
 * rest of the line is compared using the white space config for move
 * coloring. The current_longer indicates if the first string in the
 * comparision is longer than the second.
 */
struct ws_delta {
	char *string;
	unsigned int current_longer : 1;
};
#define WS_DELTA_INIT { NULL, 0 }

struct moved_block {
	struct moved_entry *match;
	struct ws_delta wsd;
};

static void moved_block_clear(struct moved_block *b)
{
	FREE_AND_NULL(b->wsd.string);
	b->match = NULL;
}

static int compute_ws_delta(const struct emitted_diff_symbol *a,
			     const struct emitted_diff_symbol *b,
			     struct ws_delta *out)
{
	const struct emitted_diff_symbol *longer =  a->len > b->len ? a : b;
	const struct emitted_diff_symbol *shorter = a->len > b->len ? b : a;
	int d = longer->len - shorter->len;

	if (strncmp(longer->line + d, shorter->line, shorter->len))
		return 0;

	out->string = xmemdupz(longer->line, d);
	out->current_longer = (a == longer);

	return 1;
