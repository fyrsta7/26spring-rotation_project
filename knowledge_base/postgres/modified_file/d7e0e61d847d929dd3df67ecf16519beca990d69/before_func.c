				}
			}
			else
			{
				if (!po->html3)
				{
					fputs(buf, fout);
			efield:
					if ((j + 1) < nFields)
						fputs(po->fieldSep, fout);
					else
						fputc('\n', fout);
				}
			}
		}
	}
}


static char *
do_header(FILE *fout, PQprintOpt *po, const int nFields, int fieldMax[],
		  char *fieldNames[], unsigned char fieldNotNum[],
		  const int fs_len, PGresult *res)
{

	int			j;				/* for loop index */
	char	   *border = NULL;

	if (po->html3)
		fputs("<tr>", fout);
	else
	{
		int			j;			/* for loop index */
		int			tot = 0;
		int			n = 0;
		char	   *p = NULL;

		for (; n < nFields; n++)
			tot += fieldMax[n] + fs_len + (po->standard ? 2 : 0);
		if (po->standard)
			tot += fs_len * 2 + 2;
		border = malloc(tot + 1);
		if (!border)
		{
			perror("malloc");
			exit(1);
		}
		p = border;
		if (po->standard)
		{
			char	   *fs = po->fieldSep;

			while (*fs++)
				*p++ = '+';
		}
		for (j = 0; j < nFields; j++)
		{
			int			len;

			for (len = fieldMax[j] + (po->standard ? 2 : 0); len--; *p++ = '-');
			if (po->standard || (j + 1) < nFields)
			{
				char	   *fs = po->fieldSep;

				while (*fs++)
					*p++ = '+';
			}
		}
		*p = '\0';
		if (po->standard)
			fprintf(fout, "%s\n", border);
	}
	if (po->standard)
		fputs(po->fieldSep, fout);
	for (j = 0; j < nFields; j++)
	{
		char	   *s = PQfname(res, j);

		if (po->html3)
		{
			fprintf(fout, "<th align=%s>%s</th>",
					fieldNotNum[j] ? "left" : "right", fieldNames[j]);
		}
		else
		{
			int			n = strlen(s);

			if (n > fieldMax[j])
				fieldMax[j] = n;
			if (po->standard)
				fprintf(fout,
						fieldNotNum[j] ? " %-*s " : " %*s ",
						fieldMax[j], s);
			else
				fprintf(fout, fieldNotNum[j] ? "%-*s" : "%*s", fieldMax[j], s);
			if (po->standard || (j + 1) < nFields)
				fputs(po->fieldSep, fout);
		}
	}
	if (po->html3)
		fputs("</tr>\n", fout);
	else
		fprintf(fout, "\n%s\n", border);
	return border;
}


static void
output_row(FILE *fout, PQprintOpt *po, const int nFields, char *fields[],
		   unsigned char fieldNotNum[], int fieldMax[], char *border,
		   const int row_index)
{

	int			field_index;	/* for loop index */

	if (po->html3)
		fputs("<tr>", fout);
	else if (po->standard)
		fputs(po->fieldSep, fout);
	for (field_index = 0; field_index < nFields; field_index++)
	{
		char	   *p = fields[row_index * nFields + field_index];

		if (po->html3)
			fprintf(fout, "<td align=%s>%s</td>",
				fieldNotNum[field_index] ? "left" : "right", p ? p : "");
		else
		{
			fprintf(fout,
					fieldNotNum[field_index] ?
					(po->standard ? " %-*s " : "%-*s") :
					(po->standard ? " %*s " : "%*s"),
					fieldMax[field_index],
					p ? p : "");
			if (po->standard || field_index + 1 < nFields)
				fputs(po->fieldSep, fout);
		}
		if (p)
			free(p);
	}
	if (po->html3)
		fputs("</tr>", fout);
	else if (po->standard)
		fprintf(fout, "\n%s", border);
	fputc('\n', fout);
}




/*
 * PQprint()
 *
 * Format results of a query for printing.
 *
 * PQprintOpt is a typedef (structure) that containes
 * various flags and options. consult libpq-fe.h for
 * details
 *
 * Obsoletes PQprintTuples.
 */

void
PQprint(FILE *fout,
		PGresult *res,
		PQprintOpt *po
)
{
	int			nFields;

	nFields = PQnfields(res);

	if (nFields > 0)
	{							/* only print rows with at least 1 field.  */
		int			i,
					j;
		int			nTups;
		int		   *fieldMax = NULL;	/* in case we don't use them */
		unsigned char *fieldNotNum = NULL;
		char	   *border = NULL;
		char	  **fields = NULL;
		char	  **fieldNames;
		int			fieldMaxLen = 0;
		int			numFieldName;
		int			fs_len = strlen(po->fieldSep);
		int			total_line_length = 0;
		int			usePipe = 0;
		char	   *pagerenv;
		char		buf[8192 * 2 + 1];

		nTups = PQntuples(res);
		if (!(fieldNames = (char **) calloc(nFields, sizeof(char *))))
		{
			perror("calloc");
			exit(1);
		}
		if (!(fieldNotNum = (unsigned char *) calloc(nFields, 1)))
		{
			perror("calloc");
			exit(1);
		}
		if (!(fieldMax = (int *) calloc(nFields, sizeof(int))))
		{
			perror("calloc");
			exit(1);
		}
		for (numFieldName = 0;
			 po->fieldName && po->fieldName[numFieldName];
			 numFieldName++)
			;
		for (j = 0; j < nFields; j++)
		{
			int			len;
			char	   *s =
			(j < numFieldName && po->fieldName[j][0]) ?
			po->fieldName[j] : PQfname(res, j);

			fieldNames[j] = s;
			len = s ? strlen(s) : 0;
			fieldMax[j] = len;
			len += fs_len;
			if (len > fieldMaxLen)
				fieldMaxLen = len;
			total_line_length += len;
		}

		total_line_length += nFields * strlen(po->fieldSep) + 1;

		if (fout == NULL)
			fout = stdout;
		if (po->pager && fout == stdout &&
			isatty(fileno(stdin)) &&
			isatty(fileno(stdout)))
		{
			/* try to pipe to the pager program if possible */
#ifdef TIOCGWINSZ
			if (ioctl(fileno(stdout), TIOCGWINSZ, &screen_size) == -1 ||
				screen_size.ws_col == 0 ||
				screen_size.ws_row == 0)
			{
#endif
				screen_size.ws_row = 24;
				screen_size.ws_col = 80;
#ifdef TIOCGWINSZ
			}
#endif
			pagerenv = getenv("PAGER");
			if (pagerenv != NULL &&
				pagerenv[0] != '\0' &&
				!po->html3 &&
				((po->expanded &&
				  nTups * (nFields + 1) >= screen_size.ws_row) ||
				 (!po->expanded &&
				  nTups * (total_line_length / screen_size.ws_col + 1) *
				  (1 + (po->standard != 0)) >=
				  screen_size.ws_row -
				  (po->header != 0) *
				  (total_line_length / screen_size.ws_col + 1) * 2
				  - (po->header != 0) * 2		/* row count and newline */
				  )))
			{
				fout = popen(pagerenv, "w");
				if (fout)
				{
					usePipe = 1;
					pqsignal(SIGPIPE, SIG_IGN);
				}
				else
					fout = stdout;
			}
		}

		if (!po->expanded && (po->align || po->html3))
		{
			if (!(fields = (char **) calloc(nFields * (nTups + 1), sizeof(char *))))
			{
				perror("calloc");
				exit(1);
			}
		}
		else if (po->header && !po->html3)
		{
			if (po->expanded)
			{
				if (po->align)
					fprintf(fout, "%-*s%s Value\n",
							fieldMaxLen - fs_len, "Field", po->fieldSep);
				else
					fprintf(fout, "%s%sValue\n", "Field", po->fieldSep);
			}
			else
			{
				int			len = 0;

				for (j = 0; j < nFields; j++)
				{
					char	   *s = fieldNames[j];

					fputs(s, fout);
					len += strlen(s) + fs_len;
					if ((j + 1) < nFields)
						fputs(po->fieldSep, fout);
				}
				fputc('\n', fout);
				for (len -= fs_len; len--; fputc('-', fout));
				fputc('\n', fout);
			}
		}
		if (po->expanded && po->html3)
		{
			if (po->caption)
				fprintf(fout, "<centre><h2>%s</h2></centre>\n", po->caption);
			else
				fprintf(fout,
						"<centre><h2>"
						"Query retrieved %d rows * %d fields"
						"</h2></centre>\n",
						nTups, nFields);
		}
		for (i = 0; i < nTups; i++)
		{
			if (po->expanded)
			{
				if (po->html3)
					fprintf(fout,
						  "<table %s><caption align=high>%d</caption>\n",
							po->tableOpt ? po->tableOpt : "", i);
				else
					fprintf(fout, "-- RECORD %d --\n", i);
			}
			for (j = 0; j < nFields; j++)
				do_field(po, res, i, j, buf, fs_len, fields, nFields,
						 fieldNames, fieldNotNum,
						 fieldMax, fieldMaxLen, fout);
			if (po->html3 && po->expanded)
				fputs("</table>\n", fout);
		}
		if (!po->expanded && (po->align || po->html3))
		{
			if (po->html3)
			{
				if (po->header)
				{
					if (po->caption)
						fprintf(fout,
						  "<table %s><caption align=high>%s</caption>\n",
								po->tableOpt ? po->tableOpt : "",
								po->caption);
					else
						fprintf(fout,
								"<table %s><caption align=high>"
								"Retrieved %d rows * %d fields"
								"</caption>\n",
						po->tableOpt ? po->tableOpt : "", nTups, nFields);
				}
				else
					fprintf(fout, "<table %s>", po->tableOpt ? po->tableOpt : "");
			}
			if (po->header)
				border = do_header(fout, po, nFields, fieldMax, fieldNames,
								   fieldNotNum, fs_len, res);
			for (i = 0; i < nTups; i++)
				output_row(fout, po, nFields, fields,
						   fieldNotNum, fieldMax, border, i);
			free(fields);
			if (border)
				free(border);
		}
		if (po->header && !po->html3)
			fprintf(fout, "(%d row%s)\n\n", PQntuples(res),
					(PQntuples(res) == 1) ? "" : "s");
		free(fieldMax);
		free(fieldNotNum);
		free(fieldNames);
		if (usePipe)
		{
			pclose(fout);
			pqsignal(SIGPIPE, SIG_DFL);
		}
		if (po->html3 && !po->expanded)
			fputs("</table>\n", fout);
	}
}


/* ----------------
 *		PQfn -	Send a function call to the POSTGRES backend.
 *
 *		conn			: backend connection
 *		fnid			: function id
 *		result_buf		: pointer to result buffer (&int if integer)
 *		result_len		: length of return value.
 *		actual_result_len: actual length returned. (differs from result_len
 *						  for varlena structures.)
 *		result_type		: If the result is an integer, this must be 1,
 *						  otherwise this should be 0
 *		args			: pointer to a NULL terminated arg array.
 *						  (length, if integer, and result-pointer)
 *		nargs			: # of arguments in args array.
 *
 * RETURNS
 *		NULL on failure.  PQerrormsg will be set.
 *		"G" if there is a return value.
 *		"V" if there is no return value.
 * ----------------
 */

PGresult   *
PQfn(PGconn *conn,
	 int fnid,
	 int *result_buf,
	 int *actual_result_len,
	 int result_is_int,
	 PQArgBlock *args,
	 int nargs)
{
	FILE	   *pfin,
			   *pfout,
			   *pfdebug;
	int			id;
	int			i;

	if (!conn)
		return NULL;

	pfin = conn->Pfin;
	pfout = conn->Pfout;
	pfdebug = conn->Pfdebug;

	/* clear the error string */
	conn->errorMessage[0] = '\0';

	pqPuts("F ", pfout, pfdebug);		/* function */
	pqPutInt(fnid, 4, pfout, pfdebug);	/* function id */
	pqPutInt(nargs, 4, pfout, pfdebug); /* # of args */

	for (i = 0; i < nargs; ++i)
	{							/* len.int4 + contents	   */
		pqPutInt(args[i].len, 4, pfout, pfdebug);
		if (args[i].isint)
		{
			pqPutInt(args[i].u.integer, 4, pfout, pfdebug);
		}
		else
		{
			pqPutnchar((char *) args[i].u.ptr, args[i].len, pfout, pfdebug);
		}
	}
	pqFlush(pfout, pfdebug);

	id = pqGetc(pfin, pfdebug);
	if (id != 'V')
	{
		if (id == 'E')
		{
			pqGets(conn->errorMessage, ERROR_MSG_LENGTH, pfin, pfdebug);
		}
		else
			sprintf(conn->errorMessage,
			   "PQfn: expected a 'V' from the backend. Got '%c' instead",
					id);
		return makeEmptyPGresult(conn, PGRES_FATAL_ERROR);
	}

	id = pqGetc(pfin, pfdebug);
	for (;;)
	{
		int			c;

		switch (id)
		{
			case 'G':			/* function returned properly */
				pqGetInt(actual_result_len, 4, pfin, pfdebug);
				if (result_is_int)
				{
					pqGetInt(result_buf, 4, pfin, pfdebug);
				}
				else
				{
					pqGetnchar((char *) result_buf, *actual_result_len,
							   pfin, pfdebug);
				}
				c = pqGetc(pfin, pfdebug);		/* get the last '0' */
				return makeEmptyPGresult(conn, PGRES_COMMAND_OK);
			case 'E':
				sprintf(conn->errorMessage,
						"PQfn: returned an error");
				return makeEmptyPGresult(conn, PGRES_FATAL_ERROR);
			case 'N':
				/* print notice and go back to processing return values */
				if (pqGets(conn->errorMessage, ERROR_MSG_LENGTH, pfin, pfdebug)
					== 1)
				{
					sprintf(conn->errorMessage,
					  "Notice return detected from backend, but message "
							"cannot be read");
				}
				else
					fprintf(stderr, "%s\n", conn->errorMessage);
				/* keep iterating */
				break;
			case '0':			/* no return value */
				return makeEmptyPGresult(conn, PGRES_COMMAND_OK);
			default:
				/* The backend violates the protocol. */
				sprintf(conn->errorMessage,
						"FATAL: PQfn: protocol error: id=%x\n", id);
				return makeEmptyPGresult(conn, PGRES_FATAL_ERROR);
		}
	}
}

/* ====== accessor funcs for PGresult ======== */

ExecStatusType
PQresultStatus(PGresult *res)
{
	if (!res)
	{
		fprintf(stderr, "PQresultStatus() -- pointer to PQresult is null");
		return PGRES_NONFATAL_ERROR;
	}

	return res->resultStatus;
}

int
PQntuples(PGresult *res)
{
	if (!res)
	{
		fprintf(stderr, "PQntuples() -- pointer to PQresult is null");
		return (int) NULL;
	}
	return res->ntups;
}

int
PQnfields(PGresult *res)
{
	if (!res)
	{
		fprintf(stderr, "PQnfields() -- pointer to PQresult is null");
		return (int) NULL;
	}
	return res->numAttributes;
}

/*
   returns NULL if the field_num is invalid
*/
char	   *
PQfname(PGresult *res, int field_num)
{
	if (!res)
	{
		fprintf(stderr, "PQfname() -- pointer to PQresult is null");
		return NULL;
	}

	if (field_num > (res->numAttributes - 1))
	{
		fprintf(stderr,
			  "PQfname: ERROR! name of field %d(of %d) is not available",
				field_num, res->numAttributes - 1);
		return NULL;
	}
	if (res->attDescs)
	{
		return res->attDescs[field_num].name;
	}
	else
		return NULL;
}

/*
   returns -1 on a bad field name
*/
int
PQfnumber(PGresult *res, const char *field_name)
{
	int			i;
	char	   *field_case;

	if (!res)
	{
		fprintf(stderr, "PQfnumber() -- pointer to PQresult is null");
		return -1;
	}

	if (field_name == NULL ||
		field_name[0] == '\0' ||
		res->attDescs == NULL)
		return -1;

	field_case = strdup(field_name);
	if (*field_case == '"')
	{
		strcpy(field_case, field_case + 1);
		*(field_case + strlen(field_case) - 1) = '\0';
	}
	else
		for (i = strlen(field_case); i >= 0; i--)
			if (isupper(field_case[i]))
				field_case[i] = tolower(field_case[i]);

	for (i = 0; i < res->numAttributes; i++)
	{
		if (strcmp(field_name, res->attDescs[i].name) == 0)
		{
			free(field_case);
			return i;
		}
	}
	free(field_case);
	return -1;
}

Oid
PQftype(PGresult *res, int field_num)
{
	if (!res)
	{
		fprintf(stderr, "PQftype() -- pointer to PQresult is null");
		return InvalidOid;
	}

	if (field_num > (res->numAttributes - 1))
	{
		fprintf(stderr,
			  "PQftype: ERROR! type of field %d(of %d) is not available",
				field_num, res->numAttributes - 1);
	}
	if (res->attDescs)
	{
		return res->attDescs[field_num].adtid;
	}
