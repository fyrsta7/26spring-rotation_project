	LPTSTR *q;

	if (!p)
		return;

	q = p;
	while (*q)
		cmd_free(*q++);

	cmd_free(p);
}


LPTSTR _stpcpy (LPTSTR dest, LPCTSTR src)
{
	_tcscpy (dest, src);
	return (dest + _tcslen (src));
}

VOID
StripQuotes(TCHAR *in)
{
	TCHAR *out = in;
	for (; *in; in++)
	{
		if (*in != _T('"'))
			*out++ = *in;
	}
	*out = _T('\0');
}



/*
 * Checks if a path is valid (accessible)
 */

BOOL IsValidPathName (LPCTSTR pszPath)
