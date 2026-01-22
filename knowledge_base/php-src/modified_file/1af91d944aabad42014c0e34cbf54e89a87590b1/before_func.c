static void zend_sort_5(void *a, void *b, void *c, void *d, void *e, compare_func_t cmp, swap_func_t swp) /* {{{ */ {
	zend_sort_4(a, b, c, d, cmp, swp);
	if (cmp(d, e) > 0) {
		swp(d, e);
		if (cmp(c, d) > 0) {
			swp(c, d);
			if (cmp(b, c) > 0) {
				swp(b, c);
				if (cmp(a, b) > 0) {
					swp(a, b);
				}
			}
		}
	}
}
/* }}} */

ZEND_API void zend_insert_sort(void *base, size_t nmemb, size_t siz, compare_func_t cmp, swap_func_t swp) /* {{{ */{
	switch (nmemb) {
		case 0:
		case 1:
			break;
		case 2:
			zend_sort_2(base, (char *)base + siz, cmp, swp);
			break;
		case 3:
			zend_sort_3(base, (char *)base + siz, (char *)base + siz + siz, cmp, swp);
			break;
		case 4:
			zend_sort_4(base, (char *)base + siz, (char *)base + siz + siz, (char *)base + siz + siz + siz, cmp, swp);
			break;
		case 5:
			zend_sort_5(base, (char *)base + siz, (char *)base + siz + siz, (char *)base + siz + siz + siz, (char *)base + (siz * 4), cmp, swp);
			break;
		default:
			{
				char *i, *j, *k;
				char *start = (char *)base;
				char *end = start + (nmemb * siz);
				char *sentry = start + (6 * siz);
				for (i = start + siz; i < sentry; i += siz) {
					j = i - siz;
					if (cmp(j, i) <= 0) {
						continue;
					}
					while (j != start) {
						j -= siz;
						if (cmp(j, i) <= 0) {
							j += siz;
							break;
						}
					}
					for (k = i; k > j; k -= siz) {
						swp(k, k - siz);
					}
				}
				for (i = sentry; i < end; i += siz) {
					j = i - siz;
					if (cmp(j, i) <= 0) {
						continue;
					}
					do {
						j -= siz * 2;
						if (cmp(j, i) <= 0) {
							j += siz;
							if (cmp(j, i) <= 0) {
								j += siz;
							}
							break;
						}
						if (j == start) {
							break;
