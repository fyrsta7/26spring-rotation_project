	size_t i, j;

	result = (char *) emalloc(oldlen * 2 * sizeof(char));
	if(!result) {
		return result;
	}
	
	for(i = j = 0; i < oldlen; i++) {
		result[j++] = hexconvtab[old[i] >> 4];
		result[j++] = hexconvtab[old[i] & 15];
	}

	if(newlen) *newlen = oldlen * 2 * sizeof(char);

	return result;
}

/* {{{ proto string bin2hex(string data)
   Converts the binary representation of data to hex */
PHP_FUNCTION(bin2hex)
