      im->colorsTotal++;
    }
  im->red[op] = r;
  im->green[op] = g;
  im->blue[op] = b;
  im->alpha[op] = a;
  im->open[op] = 0;
  return op;			/* Return newly allocated color */
}

void gdImageColorDeallocate (gdImagePtr im, int color)
{
	if (im->trueColor) {
		return;
	}
	/* Mark it open. */
	im->open[color] = 1;
}

void gdImageColorTransparent (gdImagePtr im, int color)
{
	if (!im->trueColor) {
		if (im->transparent != -1) {
			im->alpha[im->transparent] = gdAlphaOpaque;
		}
		if (color > -1 && color<im->colorsTotal && color<=gdMaxColors) {
			im->alpha[color] = gdAlphaTransparent;
		} else {
			return;
		}
	}
	im->transparent = color;
}

void gdImagePaletteCopy (gdImagePtr to, gdImagePtr from)
{
	int i;
	int x, y, p;
