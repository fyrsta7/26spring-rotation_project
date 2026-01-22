			appendStringInfo(buf, "FOR TYPE %s", format_type_be(trftypes[i]));
		}
		appendStringInfoChar(buf, '\n');
	}
}

/*
 * Get textual representation of a function argument's default value.  The
 * second argument of this function is the argument number among all arguments
 * (i.e. proallargtypes, *not* proargtypes), starting with 1, because that's
 * how information_schema.sql uses it.
 */
Datum
pg_get_function_arg_default(PG_FUNCTION_ARGS)
{
	Oid			funcid = PG_GETARG_OID(0);
	int32		nth_arg = PG_GETARG_INT32(1);
	HeapTuple	proctup;
	Form_pg_proc proc;
	int			numargs;
	Oid		   *argtypes;
	char	  **argnames;
