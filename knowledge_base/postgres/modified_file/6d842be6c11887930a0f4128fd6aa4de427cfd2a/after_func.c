	len = pg_vsprintf(str, fmt, args);
	va_end(args);
	return len;
}

int
pg_vfprintf(FILE *stream, const char *fmt, va_list args)
{
	PrintfTarget target;
	char		buffer[1024];	/* size is arbitrary */

	if (stream == NULL)
	{
		errno = EINVAL;
		return -1;
	}
	target.bufstart = target.bufptr = buffer;
	target.bufend = buffer + sizeof(buffer);	/* use the whole buffer */
	target.stream = stream;
	target.nchars = 0;
	target.failed = false;
	dopr(&target, fmt, args);
	/* dump any remaining buffer contents */
	flushbuffer(&target);
	return target.failed ? -1 : target.nchars;
}

int
pg_fprintf(FILE *stream, const char *fmt,...)
{
	int			len;
	va_list		args;

	va_start(args, fmt);
	len = pg_vfprintf(stream, fmt, args);
	va_end(args);
	return len;
}

int
pg_printf(const char *fmt,...)
{
	int			len;
	va_list		args;

	va_start(args, fmt);
	len = pg_vfprintf(stdout, fmt, args);
	va_end(args);
	return len;
}

/*
 * Attempt to write the entire buffer to target->stream; discard the entire
 * buffer in any case.  Call this only when target->stream is defined.
 */
static void
flushbuffer(PrintfTarget *target)
{
	size_t		nc = target->bufptr - target->bufstart;

	/*
	 * Don't write anything if we already failed; this is to ensure we
	 * preserve the original failure's errno.
	 */
	if (!target->failed && nc > 0)
	{
		size_t		written;

		written = fwrite(target->bufstart, 1, nc, target->stream);
		target->nchars += written;
		if (written != nc)
			target->failed = true;
	}
	target->bufptr = target->bufstart;
}


static bool find_arguments(const char *format, va_list args,
			   PrintfArgValue *argvalues);
static void fmtstr(const char *value, int leftjust, int minlen, int maxwidth,
	   int pointflag, PrintfTarget *target);
static void fmtptr(void *value, PrintfTarget *target);
static void fmtint(int64 value, char type, int forcesign,
	   int leftjust, int minlen, int zpad, int precision, int pointflag,
	   PrintfTarget *target);
static void fmtchar(int value, int leftjust, int minlen, PrintfTarget *target);
static void fmtfloat(double value, char type, int forcesign,
		 int leftjust, int minlen, int zpad, int precision, int pointflag,
		 PrintfTarget *target);
static void dostr(const char *str, int slen, PrintfTarget *target);
static void dopr_outch(int c, PrintfTarget *target);
static void dopr_outchmulti(int c, int slen, PrintfTarget *target);
static int	adjust_sign(int is_negative, int forcesign, int *signvalue);
static int	compute_padlen(int minlen, int vallen, int leftjust);
static void leading_pad(int zpad, int signvalue, int *padlen,
			PrintfTarget *target);
static void trailing_pad(int padlen, PrintfTarget *target);

/*
 * If strchrnul exists (it's a glibc-ism), it's a good bit faster than the
 * equivalent manual loop.  If it doesn't exist, provide a replacement.
 *
 * Note: glibc declares this as returning "char *", but that would require
 * casting away const internally, so we don't follow that detail.
 */
#ifndef HAVE_STRCHRNUL

static inline const char *
strchrnul(const char *s, int c)
{
	while (*s != '\0' && *s != c)
		s++;
	return s;
}

#else

/*
 * glibc's <string.h> declares strchrnul only if _GNU_SOURCE is defined.
 * While we typically use that on glibc platforms, configure will set
 * HAVE_STRCHRNUL whether it's used or not.  Fill in the missing declaration
 * so that this file will compile cleanly with or without _GNU_SOURCE.
 */
#ifndef _GNU_SOURCE
extern char *strchrnul(const char *s, int c);
#endif

#endif							/* HAVE_STRCHRNUL */


/*
 * dopr(): the guts of *printf for all cases.
 */
static void
dopr(PrintfTarget *target, const char *format, va_list args)
{
	int			save_errno = errno;
	const char *first_pct = NULL;
	int			ch;
	bool		have_dollar;
	bool		have_star;
	bool		afterstar;
	int			accum;
	int			longlongflag;
	int			longflag;
	int			pointflag;
	int			leftjust;
	int			fieldwidth;
	int			precision;
	int			zpad;
	int			forcesign;
	int			fmtpos;
	int			cvalue;
	int64		numvalue;
	double		fvalue;
	char	   *strvalue;
	PrintfArgValue argvalues[PG_NL_ARGMAX + 1];

	/*
	 * Initially, we suppose the format string does not use %n$.  The first
	 * time we come to a conversion spec that has that, we'll call
	 * find_arguments() to check for consistent use of %n$ and fill the
	 * argvalues array with the argument values in the correct order.
	 */
	have_dollar = false;

	while (*format != '\0')
	{
		/* Locate next conversion specifier */
		if (*format != '%')
		{
			/* Scan to next '%' or end of string */
			const char *next_pct = strchrnul(format + 1, '%');

			/* Dump literal data we just scanned over */
			dostr(format, next_pct - format, target);
			if (target->failed)
				break;

			if (*next_pct == '\0')
				break;
			format = next_pct;
		}

		/*
		 * Remember start of first conversion spec; if we find %n$, then it's
		 * sufficient for find_arguments() to start here, without rescanning
		 * earlier literal text.
		 */
		if (first_pct == NULL)
			first_pct = format;

		/* Process conversion spec starting at *format */
		format++;

		/* Fast path for conversion spec that is exactly %s */
		if (*format == 's')
		{
			format++;
			strvalue = va_arg(args, char *);
			Assert(strvalue != NULL);
			dostr(strvalue, strlen(strvalue), target);
			if (target->failed)
				break;
			continue;
		}

		fieldwidth = precision = zpad = leftjust = forcesign = 0;
		longflag = longlongflag = pointflag = 0;
		fmtpos = accum = 0;
		have_star = afterstar = false;
nextch2:
		ch = *format++;
		if (ch == '\0')
			break;				/* illegal, but we don't complain */
		switch (ch)
		{
			case '-':
				leftjust = 1;
				goto nextch2;
			case '+':
				forcesign = 1;
				goto nextch2;
			case '0':
				/* set zero padding if no nonzero digits yet */
				if (accum == 0 && !pointflag)
					zpad = '0';
				/* FALL THRU */
			case '1':
			case '2':
			case '3':
			case '4':
			case '5':
			case '6':
			case '7':
			case '8':
			case '9':
				accum = accum * 10 + (ch - '0');
				goto nextch2;
			case '.':
				if (have_star)
					have_star = false;
				else
					fieldwidth = accum;
				pointflag = 1;
				accum = 0;
				goto nextch2;
			case '*':
				if (have_dollar)
				{
					/*
					 * We'll process value after reading n$.  Note it's OK to
					 * assume have_dollar is set correctly, because in a valid
					 * format string the initial % must have had n$ if * does.
					 */
					afterstar = true;
				}
				else
				{
					/* fetch and process value now */
					int			starval = va_arg(args, int);

					if (pointflag)
					{
						precision = starval;
						if (precision < 0)
						{
							precision = 0;
							pointflag = 0;
						}
					}
					else
					{
						fieldwidth = starval;
						if (fieldwidth < 0)
						{
							leftjust = 1;
							fieldwidth = -fieldwidth;
						}
					}
				}
				have_star = true;
				accum = 0;
				goto nextch2;
			case '$':
				/* First dollar sign? */
				if (!have_dollar)
				{
					/* Yup, so examine all conversion specs in format */
					if (!find_arguments(first_pct, args, argvalues))
						goto bad_format;
					have_dollar = true;
				}
				if (afterstar)
				{
					/* fetch and process star value */
					int			starval = argvalues[accum].i;

					if (pointflag)
					{
						precision = starval;
						if (precision < 0)
						{
							precision = 0;
							pointflag = 0;
						}
					}
					else
					{
						fieldwidth = starval;
						if (fieldwidth < 0)
						{
							leftjust = 1;
							fieldwidth = -fieldwidth;
						}
					}
					afterstar = false;
				}
				else
					fmtpos = accum;
				accum = 0;
				goto nextch2;
			case 'l':
				if (longflag)
					longlongflag = 1;
				else
					longflag = 1;
				goto nextch2;
			case 'z':
#if SIZEOF_SIZE_T == 8
#ifdef HAVE_LONG_INT_64
				longflag = 1;
#elif defined(HAVE_LONG_LONG_INT_64)
				longlongflag = 1;
#else
#error "Don't know how to print 64bit integers"
#endif
#else
				/* assume size_t is same size as int */
#endif
				goto nextch2;
			case 'h':
			case '\'':
				/* ignore these */
				goto nextch2;
			case 'd':
