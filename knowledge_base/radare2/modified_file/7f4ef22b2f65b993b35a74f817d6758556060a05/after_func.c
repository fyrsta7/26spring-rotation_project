}

// Returns the string representation of the permission of the inputted integer.
R_API const char *r_str_rwx_i(int rwx) {
	if (rwx < 0 || rwx >= R_ARRAY_SIZE (rwxstr)) {
		rwx = 0;
