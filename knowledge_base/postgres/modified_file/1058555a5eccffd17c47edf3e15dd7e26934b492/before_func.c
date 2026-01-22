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

	if (*txt == '\0' || searchstoplist(&(d->stoplist), txt))
	{
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
				pfree(txt);
				txt = recoded;
			}
		}

		/* see comment about d->dictCtx */
		saveCtx = MemoryContextSwitchTo(d->dictCtx);
		SN_set_current(d->z, strlen(txt), (symbol *) txt);
		d->stem(d->z);
