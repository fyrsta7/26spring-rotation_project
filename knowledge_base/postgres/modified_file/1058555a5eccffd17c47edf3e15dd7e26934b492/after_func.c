			if (d->stem)
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("multiple Language parameters")));
			locate_stem_module(d, defGetString(defel));
		}
		else
		{
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("unrecognized Snowball parameter: \"%s\"",
							defel->defname)));
		}
	}

	if (!d->stem)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("missing Language parameter")));

	d->dictCtx = CurrentMemoryContext;

	PG_RETURN_POINTER(d);
}

Datum
dsnowball_lexize(PG_FUNCTION_ARGS)
{
	DictSnowball *d = (DictSnowball *) PG_GETARG_POINTER(0);
	char	   *in = (char *) PG_GETARG_POINTER(1);
	int32		len = PG_GETARG_INT32(2);
	char	   *txt = lowerstr_with_len(in, len);
	TSLexeme   *res = palloc0(sizeof(TSLexeme) * 2);

	/*
	 * Do not pass strings exceeding 1000 bytes to the stemmer, as they're
	 * surely not words in any human language.  This restriction avoids
	 * wasting cycles on stuff like base64-encoded data, and it protects us
	 * against possible inefficiency or misbehavior in the stemmer.  (For
	 * example, the Turkish stemmer has an indefinite recursion, so it can
	 * crash on long-enough strings.)  However, Snowball dictionaries are
	 * defined to recognize all strings, so we can't reject the string as an
	 * unknown word.
	 */
	if (len > 1000)
	{
		/* return the lexeme lowercased, but otherwise unmodified */
		res->lexeme = txt;
	}
	else if (*txt == '\0' || searchstoplist(&(d->stoplist), txt))
	{
		/* empty or stopword, so report as stopword */
		pfree(txt);
	}
	else
	{
		MemoryContext saveCtx;

		/*
		 * recode to utf8 if stemmer is utf8 and doesn't match server encoding
		 */
		if (d->needrecode)
		{
			char	   *recoded;

			recoded = pg_server_to_any(txt, strlen(txt), PG_UTF8);
			if (recoded != txt)
			{
