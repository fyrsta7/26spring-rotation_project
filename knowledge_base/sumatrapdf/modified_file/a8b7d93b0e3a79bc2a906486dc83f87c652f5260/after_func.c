
static void
fz_stdconvpixmap(fz_pixmap *src, fz_pixmap *dst)
{
	float srcv[FZ_MAXCOLORS];
	float dstv[FZ_MAXCOLORS];
	int y, x, k;

	fz_colorspace *ss = src->colorspace;
	fz_colorspace *ds = dst->colorspace;

	unsigned char *s = src->samples;
	unsigned char *d = dst->samples;

	/* cf. http://bugs.ghostscript.com/show_bug.cgi?id=691637 */
	/* memoize fz_convertcolor, as it can be quite inefficient for PostScript functions */
	fz_hashtable *colors = fz_newhash(509, src->n - 1);
	unsigned char *c;

	assert(src->w == dst->w && src->h == dst->h);
	assert(src->n == ss->n + 1);
	assert(dst->n == ds->n + 1);

	for (y = 0; y < src->h; y++)
	{
		for (x = 0; x < src->w; x++)
		{
			/* cf. http://bugs.ghostscript.com/show_bug.cgi?id=691637 */
			if ((c = fz_hashfind(colors, s)))
			{
				memcpy(d, c, dst->n - 1);
				s += src->n - 1;
				d += dst->n - 1;
				*d++ = *s++;
				continue;
			}

			for (k = 0; k < src->n - 1; k++)
				srcv[k] = *s++ / 255.0f;

			fz_convertcolor(ss, srcv, ds, dstv);

			for (k = 0; k < dst->n - 1; k++)
				*d++ = dstv[k] * 255;

			*d++ = *s++;

			fz_hashinsert(colors, s - src->n, d - dst->n);
		}
	}

	fz_freehash(colors);
