	   requested encoding, but avoids setting LC_CTYPE from the
	   environment for the whole program.

	   This primarily done to avoid a bug in vsnprintf in the GNU C
	   Library [1]. which triggered a "your vsnprintf is broken" error
	   on Git's own repository when inspecting v0.99.6~1 under a UTF-8
	   locale.

	   That commit contains a ISO-8859-1 encoded author name, which
	   the locale aware vsnprintf(3) won't interpolate in the format
	   argument, due to mismatch between the data encoding and the
	   locale.

