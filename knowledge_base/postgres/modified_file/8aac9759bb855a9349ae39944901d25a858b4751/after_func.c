			}
		}
	}
	PG_RETURN_POINTER(result);
}

Datum
g_intbig_compress(PG_FUNCTION_ARGS)
{
	GISTENTRY  *entry = (GISTENTRY *) PG_GETARG_POINTER(0);
	int			siglen = GET_SIGLEN();

	if (entry->leafkey)
	{
		GISTENTRY  *retval;
		ArrayType  *in = DatumGetArrayTypeP(entry->key);
		int32	   *ptr;
		int			num;
		GISTTYPE   *res = _intbig_alloc(false, siglen, NULL);

		CHECKARRVALID(in);
		if (ARRISEMPTY(in))
		{
			ptr = NULL;
			num = 0;
		}
		else
		{
			ptr = ARRPTR(in);
			num = ARRNELEMS(in);
		}

		while (num--)
		{
			HASH(GETSIGN(res), *ptr, siglen);
			ptr++;
		}

		retval = (GISTENTRY *) palloc(sizeof(GISTENTRY));
		gistentryinit(*retval, PointerGetDatum(res),
					  entry->rel, entry->page,
					  entry->offset, false);

		PG_RETURN_POINTER(retval);
	}
	else if (!ISALLTRUE(DatumGetPointer(entry->key)))
	{
		GISTENTRY  *retval;
		int			i;
		BITVECP		sign = GETSIGN(DatumGetPointer(entry->key));
		GISTTYPE   *res;

		LOOPBYTE(siglen)
		{
			if ((sign[i] & 0xff) != 0xff)
				PG_RETURN_POINTER(entry);
		}

		res = _intbig_alloc(true, siglen, sign);
		retval = (GISTENTRY *) palloc(sizeof(GISTENTRY));
		gistentryinit(*retval, PointerGetDatum(res),
					  entry->rel, entry->page,
					  entry->offset, false);
