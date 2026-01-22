PHPAPI char *php_addslashes(char *str, int length, int *new_length, int freeit);
PHPAPI char *php_addcslashes(char *str, int length, int *new_length, int freeit, char *what, int wlength);
PHPAPI void php_stripslashes(char *str, int *len);
PHPAPI void php_stripcslashes(char *str, int *len);
PHPAPI char *php_basename(char *str, size_t  len ,char *suffix, size_t sufflen);
PHPAPI void php_dirname(char *str, int len);
PHPAPI char *php_stristr(unsigned char *s, unsigned char *t, size_t s_len, size_t t_len);
PHPAPI char *php_str_to_str(char *haystack, int length, char *needle,
		int needle_len, char *str, int str_len, int *_new_length);
PHPAPI void php_trim(pval *str, pval *return_value, int mode);
PHPAPI void php_strip_tags(char *rbuf, int len, int state, char *allow, int allow_len);

PHPAPI void php_char_to_str(char *str, uint len, char from, char *to, int to_len, pval *result);

PHPAPI void php_implode(pval *delim, pval *arr, pval *return_value);
PHPAPI void php_explode(pval *delim, pval *str, pval *return_value, int limit);

static inline char *
php_memnstr(char *haystack, char *needle, int needle_len, char *end)
