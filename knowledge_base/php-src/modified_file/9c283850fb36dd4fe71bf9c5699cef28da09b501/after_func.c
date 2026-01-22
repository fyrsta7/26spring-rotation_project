			s = 0xA1FE;
		}
	}

	if (s <= 0) {
		if (c == 0) {
			s = 0;
		} else {
			s = -1;
		}
	}

	if (s >= 0) {
		if (s <= 0x80) { /* latin */
			CK((*filter->output_function)(s, filter->data));
		} else {
			CK((*filter->output_function)((s >> 8) & 0xff, filter->data));
			CK((*filter->output_function)(s & 0xff, filter->data));
		}
	} else {
		CK(mbfl_filt_conv_illegal_output(c, filter));
	}

	return 0;
}

static size_t mb_big5_to_wchar(unsigned char **in, size_t *in_len, uint32_t *buf, size_t bufsize, unsigned int *state)
{
	unsigned char *p = *in, *e = p + *in_len;
	uint32_t *out = buf, *limit = buf + bufsize;

	e--; /* Stop the main loop 1 byte short of the end of the input */

	while (p < e && out < limit) {
		unsigned char c = *p++;

		if (c <= 0x7F) {
			*out++ = c;
		} else if (c > 0xA0 && c <= 0xF9 && c != 0xC8) {
			/* We don't need to check p < e here; it's not possible that this pointer dereference
			 * will be outside the input string, because of e-- above */
