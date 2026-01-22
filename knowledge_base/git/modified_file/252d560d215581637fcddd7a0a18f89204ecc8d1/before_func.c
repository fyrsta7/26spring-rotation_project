			bol = pmatch[0].rm_so + bol + 1;
			while (word_char(bol[-1]) && bol < eol)
				bol++;
			if (bol < eol)
				goto again;
		}
	}
	if (p->token == GREP_PATTERN_HEAD && saved_ch)
		*eol = saved_ch;
	return hit;
}

static int match_expr_eval(struct grep_opt *o,
			   struct grep_expr *x,
			   char *bol, char *eol,
			   enum grep_context ctx,
			   int collect_hits)
{
	int h = 0;

	switch (x->node) {
	case GREP_NODE_ATOM:
		h = match_one_pattern(o, x->u.atom, bol, eol, ctx);
		break;
	case GREP_NODE_NOT:
		h = !match_expr_eval(o, x->u.unary, bol, eol, ctx, 0);
		break;
	case GREP_NODE_AND:
		if (!collect_hits)
			return (match_expr_eval(o, x->u.binary.left,
						bol, eol, ctx, 0) &&
				match_expr_eval(o, x->u.binary.right,
						bol, eol, ctx, 0));
		h = match_expr_eval(o, x->u.binary.left, bol, eol, ctx, 0);
		h &= match_expr_eval(o, x->u.binary.right, bol, eol, ctx, 0);
		break;
	case GREP_NODE_OR:
		if (!collect_hits)
			return (match_expr_eval(o, x->u.binary.left,
						bol, eol, ctx, 0) ||
				match_expr_eval(o, x->u.binary.right,
