	ZEND_ASSERT(string);

	mbfl_memory_device_realloc(&convd->device, convd->device.pos + string->len, string->len/4);
	/* feed data */
	n = string->len;
	p = string->val;

	filter = convd->filter1;
	if (filter != NULL) {
		while (n > 0) {
			if ((*filter->filter_function)(*p++, filter) < 0) {
				return p - string->val;
			}
			n--;
		}
	}
