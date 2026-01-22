	do { \
		int st = snprintf(s, len, "%llu", i); \
		s[st] = '\0'; \
	} while (0)
#endif
#define PHP_RETURN_HRTIME(t) do { \
	char _a[ZEND_LTOA_BUF_LEN]; \
	double _d; \
	HRTIME_U64A(t, _a, ZEND_LTOA_BUF_LEN); \
	_d = zend_strtod(_a, NULL); \
	RETURN_DOUBLE(_d); \
	} while (0)
#endif

/* {{{ Returns an array of integers in form [seconds, nanoseconds] counted
	from an arbitrary point in time. If an optional boolean argument is
	passed, returns an integer on 64-bit platforms or float on 32-bit
	containing the current high-resolution time in nanoseconds. The
	delivered timestamp is monotonic and cannot be adjusted. */
PHP_FUNCTION(hrtime)
{
#if ZEND_HRTIME_AVAILABLE
	bool get_as_num = 0;
