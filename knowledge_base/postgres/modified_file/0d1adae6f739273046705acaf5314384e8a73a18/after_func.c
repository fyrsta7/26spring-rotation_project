	PG_RETURN_BYTEA_P(pq_endtypsend(&buf));
}

/*
 * Binary receive.
 */
Datum
json_recv(PG_FUNCTION_ARGS)
{
	StringInfo	buf = (StringInfo) PG_GETARG_POINTER(0);
	char	   *str;
	int			nbytes;
	JsonLexContext lex;

	str = pq_getmsgtext(buf, buf->len - buf->cursor, &nbytes);

	/* Validate it. */
	makeJsonLexContextCstringLen(&lex, str, nbytes, GetDatabaseEncoding(),
								 false);
	pg_parse_json_or_ereport(&lex, &nullSemAction);

	PG_RETURN_TEXT_P(cstring_to_text_with_len(str, nbytes));
}

/*
 * Turn a Datum into JSON text, appending the string to "result".
 *
 * tcategory and outfuncoid are from a previous call to json_categorize_type,
 * except that if is_null is true then they can be invalid.
 *
 * If key_scalar is true, the value is being printed as a key, so insist
 * it's of an acceptable type, and force it to be quoted.
 */
static void
datum_to_json_internal(Datum val, bool is_null, StringInfo result,
					   JsonTypeCategory tcategory, Oid outfuncoid,
					   bool key_scalar)
{
	char	   *outputstr;
	text	   *jsontext;

	check_stack_depth();

	/* callers are expected to ensure that null keys are not passed in */
	Assert(!(key_scalar && is_null));

	if (is_null)
	{
		appendBinaryStringInfo(result, "null", strlen("null"));
		return;
	}

	if (key_scalar &&
		(tcategory == JSONTYPE_ARRAY ||
		 tcategory == JSONTYPE_COMPOSITE ||
		 tcategory == JSONTYPE_JSON ||
		 tcategory == JSONTYPE_CAST))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("key value must be scalar, not array, composite, or json")));

	switch (tcategory)
	{
		case JSONTYPE_ARRAY:
			array_to_json_internal(val, result, false);
			break;
		case JSONTYPE_COMPOSITE:
			composite_to_json(val, result, false);
			break;
		case JSONTYPE_BOOL:
			if (key_scalar)
				appendStringInfoChar(result, '"');
			if (DatumGetBool(val))
				appendBinaryStringInfo(result, "true", strlen("true"));
			else
				appendBinaryStringInfo(result, "false", strlen("false"));
			if (key_scalar)
				appendStringInfoChar(result, '"');
			break;
		case JSONTYPE_NUMERIC:
			outputstr = OidOutputFunctionCall(outfuncoid, val);

			/*
			 * Don't quote a non-key if it's a valid JSON number (i.e., not
			 * "Infinity", "-Infinity", or "NaN").  Since we know this is a
			 * numeric data type's output, we simplify and open-code the
			 * validation for better performance.
			 */
			if (!key_scalar &&
				((*outputstr >= '0' && *outputstr <= '9') ||
				 (*outputstr == '-' &&
				  (outputstr[1] >= '0' && outputstr[1] <= '9'))))
				appendStringInfoString(result, outputstr);
			else
			{
				appendStringInfoChar(result, '"');
				appendStringInfoString(result, outputstr);
				appendStringInfoChar(result, '"');
			}
			pfree(outputstr);
			break;
		case JSONTYPE_DATE:
			{
				char		buf[MAXDATELEN + 1];

				JsonEncodeDateTime(buf, val, DATEOID, NULL);
				appendStringInfoChar(result, '"');
				appendStringInfoString(result, buf);
				appendStringInfoChar(result, '"');
			}
			break;
		case JSONTYPE_TIMESTAMP:
			{
