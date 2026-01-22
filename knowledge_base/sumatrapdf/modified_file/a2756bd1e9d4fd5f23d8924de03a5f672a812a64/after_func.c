merge_lines(fz_context *ctx, fz_text_block *block, fz_text_line *line)
{
	if (line == block->lines + block->len - 1)
		return;
	while ((line + 1)->first_span)
	{
		fz_union_rect(&line->bbox, &(line + 1)->first_span->bbox);
		line->last_span = line->last_span->next = (line + 1)->first_span;
		(line + 1)->first_span = (line + 1)->first_span->next;
	}
	memmove(line + 1, line + 2, (block->lines + block->len - (line + 2)) * sizeof(fz_text_line));
	block->len--;
}

static void
fixup_text_block(fz_context *ctx, fz_text_block *block)
{
	fz_text_line *line;
	fz_text_span *span;
	int i;

	/* cf. http://code.google.com/p/sumatrapdf/issues/detail?id=734 */
	/* remove duplicate character sequences in (almost) the same spot */
	for (line = block->lines; line < block->lines + block->len; line++)
	{
		for (span = line->first_span; span; span = span->next)
		{
			for (i = 0; i < span->len; i++)
			{
				fz_text_line *line2 = line;
				fz_text_span *span2 = span;
				int j = i + 1;
				for (;;)
				{
					if (!span2)
					{
						if (line2 + 1 == block->lines + block->len || line2 != line || !(line2 + 1)->first_span)
							break;
						line2++;
						span2 = line2->first_span;
					}
					for (; j < span2->len && j < i + 512; j++)
					{
						int c = span->text[i].c;
						if (c != 32 && c == span2->text[j].c && do_glyphs_overlap(span, i, span2, j, 1))
							goto fixup_delete_duplicates;
					}
					span2 = span2->next;
					j = 0;
				}
				continue;

fixup_delete_duplicates:
				do
