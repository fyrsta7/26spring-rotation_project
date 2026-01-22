
int mbfl_filt_conv_wchar_utf8(int c, mbfl_convert_filter *filter)
{
	if (c >= 0 && c < 0x110000) {
		if (c < 0x80) {
			CK((*filter->output_function)(c, filter->data));
		} else if (c < 0x800) {
			CK((*filter->output_function)(((c >> 6) & 0x1f) | 0xc0, filter->data));
			CK((*filter->output_function)((c & 0x3f) | 0x80, filter->data));
		} else if (c < 0x10000) {
			CK((*filter->output_function)(((c >> 12) & 0x0f) | 0xe0, filter->data));
			CK((*filter->output_function)(((c >> 6) & 0x3f) | 0x80, filter->data));
			CK((*filter->output_function)((c & 0x3f) | 0x80, filter->data));
		} else {
			CK((*filter->output_function)(((c >> 18) & 0x07) | 0xf0, filter->data));
			CK((*filter->output_function)(((c >> 12) & 0x3f) | 0x80, filter->data));
			CK((*filter->output_function)(((c >> 6) & 0x3f) | 0x80, filter->data));
			CK((*filter->output_function)((c & 0x3f) | 0x80, filter->data));
		}
	} else {
		CK(mbfl_filt_conv_illegal_output(c, filter));
	}

	return 0;
}

static size_t mb_utf8_to_wchar(unsigned char **in, size_t *in_len, uint32_t *buf, size_t bufsize, unsigned int *state)
{
	unsigned char *p = *in, *e = p + *in_len;
	uint32_t *out = buf, *limit = buf + bufsize;

	while (p < e && out < limit) {
		unsigned char c = *p++;

		if (c < 0x80) {
			*out++ = c;
		} else if (c >= 0xC2 && c <= 0xDF) { /* 2 byte character */
			if (p < e) {
				unsigned char c2 = *p++;
				if ((c2 & 0xC0) != 0x80) {
					*out++ = MBFL_BAD_INPUT;
					p--;
				} else {
					*out++ = ((c & 0x1F) << 6) | (c2 & 0x3F);
				}
			} else {
				*out++ = MBFL_BAD_INPUT;
			}
		} else if (c >= 0xE0 && c <= 0xEF) { /* 3 byte character */
			if ((e - p) >= 2) {
				unsigned char c2 = *p++;
				unsigned char c3 = *p++;
				if ((c2 & 0xC0) != 0x80 || (c == 0xE0 && c2 < 0xA0) || (c == 0xED && c2 >= 0xA0)) {
					*out++ = MBFL_BAD_INPUT;
					p -= 2;
				} else if ((c3 & 0xC0) != 0x80) {
					*out++ = MBFL_BAD_INPUT;
					p--;
				} else {
					uint32_t decoded = ((c & 0xF) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);
					ZEND_ASSERT(decoded >= 0x800); /* Not an overlong code unit */
					ZEND_ASSERT(decoded < 0xD800 || decoded > 0xDFFF); /* U+D800-DFFF are reserved, illegal code points */
					*out++ = decoded;
				}
			} else {
				*out++ = MBFL_BAD_INPUT;
				if (p < e && (c != 0xE0 || *p >= 0xA0) && (c != 0xED || *p < 0xA0) && (*p & 0xC0) == 0x80) {
					p++;
					if (p < e && (*p & 0xC0) == 0x80) {
						p++;
					}
				}
			}
		} else if (c >= 0xF0 && c <= 0xF4) { /* 4 byte character */
			if ((e - p) >= 3) {
				unsigned char c2 = *p++;
				unsigned char c3 = *p++;
				unsigned char c4 = *p++;
				/* If c == 0xF0 and c2 < 0x90, then this is an over-long code unit; it could have
				 * fit in 3 bytes only. If c == 0xF4 and c2 >= 0x90, then this codepoint is
				 * greater than U+10FFFF, which is the highest legal codepoint */
				if ((c2 & 0xC0) != 0x80 || (c == 0xF0 && c2 < 0x90) || (c == 0xF4 && c2 >= 0x90)) {
					*out++ = MBFL_BAD_INPUT;
					p -= 3;
				} else if ((c3 & 0xC0) != 0x80) {
					*out++ = MBFL_BAD_INPUT;
					p -= 2;
