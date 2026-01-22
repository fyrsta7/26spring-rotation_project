	char		command[SHELL_COMMAND_SIZE];
	int			i,
				len = 0;
	FILE	   *fp;
	char		res[64];
	char	   *endptr;
	int			retval;

	/*----------
	 * Join arguments with whitespace separators. Arguments starting with
	 * exactly one colon are treated as variables:
	 *	name - append a string "name"
	 *	:var - append a variable named 'var'
	 *	::name - append a string ":name"
	 *----------
	 */
	for (i = 0; i < argc; i++)
	{
		char	   *arg;
		int			arglen;

		if (argv[i][0] != ':')
		{
			arg = argv[i];		/* a string literal */
		}
		else if (argv[i][1] == ':')
		{
			arg = argv[i] + 1;	/* a string literal starting with colons */
		}
		else if ((arg = getVariable(st, argv[i] + 1)) == NULL)
		{
			pg_log_error("%s: undefined variable \"%s\"", argv[0], argv[i]);
			return false;
		}

		arglen = strlen(arg);
		if (len + arglen + (i > 0 ? 1 : 0) >= SHELL_COMMAND_SIZE - 1)
		{
			pg_log_error("%s: shell command is too long", argv[0]);
			return false;
		}

		if (i > 0)
			command[len++] = ' ';
		memcpy(command + len, arg, arglen);
		len += arglen;
	}

	command[len] = '\0';

	/* Fast path for non-assignment case */
	if (variable == NULL)
	{
		if (system(command))
		{
			if (!timer_exceeded)
				pg_log_error("%s: could not launch shell command", argv[0]);
			return false;
		}
		return true;
	}

	/* Execute the command with pipe and read the standard output. */
	if ((fp = popen(command, "r")) == NULL)
	{
		pg_log_error("%s: could not launch shell command", argv[0]);
		return false;
	}
	if (fgets(res, sizeof(res), fp) == NULL)
	{
		if (!timer_exceeded)
			pg_log_error("%s: could not read result of shell command", argv[0]);
		(void) pclose(fp);
		return false;
	}
	if (pclose(fp) < 0)
	{
		pg_log_error("%s: could not close shell command", argv[0]);
		return false;
	}

	/* Check whether the result is an integer and assign it to the variable */
	retval = (int) strtol(res, &endptr, 10);
	while (*endptr != '\0' && isspace((unsigned char) *endptr))
		endptr++;
	if (*res == '\0' || *endptr != '\0')
	{
		pg_log_error("%s: shell command must return an integer (not \"%s\")", argv[0], res);
		return false;
	}
	if (!putVariableInt(st, "setshell", variable, retval))
		return false;

	pg_log_debug("%s: shell parameter name: \"%s\", value: \"%s\"", argv[0], argv[1], res);

	return true;
}

#define MAX_PREPARE_NAME		32
static void
preparedStatementName(char *buffer, int file, int state)
{
	sprintf(buffer, "P%d_%d", file, state);
}

static void
commandFailed(CState *st, const char *cmd, const char *message)
{
	pg_log_error("client %d aborted in command %d (%s) of script %d; %s",
				 st->id, st->command, cmd, st->use_file, message);
}

/* return a script number with a weighted choice. */
static int
chooseScript(TState *thread)
{
	int			i = 0;
	int64		w;

	if (num_scripts == 1)
		return 0;

	w = getrand(&thread->ts_choose_rs, 0, total_weight - 1);
	do
	{
		w -= sql_script[i++].weight;
	} while (w >= 0);

	return i - 1;
}

/* Send a SQL command, using the chosen querymode */
static bool
sendCommand(CState *st, Command *command)
{
	int			r;

	if (querymode == QUERY_SIMPLE)
	{
		char	   *sql;

		sql = pg_strdup(command->argv[0]);
		sql = assignVariables(st, sql);

		pg_log_debug("client %d sending %s", st->id, sql);
		r = PQsendQuery(st->con, sql);
		free(sql);
	}
	else if (querymode == QUERY_EXTENDED)
	{
		const char *sql = command->argv[0];
		const char *params[MAX_ARGS];

		getQueryParams(st, command, params);

		pg_log_debug("client %d sending %s", st->id, sql);
		r = PQsendQueryParams(st->con, sql, command->argc - 1,
							  NULL, params, NULL, NULL, 0);
	}
	else if (querymode == QUERY_PREPARED)
	{
		char		name[MAX_PREPARE_NAME];
		const char *params[MAX_ARGS];

		if (!st->prepared[st->use_file])
		{
			int			j;
			Command   **commands = sql_script[st->use_file].commands;

			for (j = 0; commands[j] != NULL; j++)
			{
				PGresult   *res;
				char		name[MAX_PREPARE_NAME];

				if (commands[j]->type != SQL_COMMAND)
					continue;
				preparedStatementName(name, st->use_file, j);
				if (PQpipelineStatus(st->con) == PQ_PIPELINE_OFF)
				{
					res = PQprepare(st->con, name,
									commands[j]->argv[0], commands[j]->argc - 1, NULL);
					if (PQresultStatus(res) != PGRES_COMMAND_OK)
						pg_log_error("%s", PQerrorMessage(st->con));
					PQclear(res);
				}
				else
				{
					/*
					 * In pipeline mode, we use asynchronous functions. If a
					 * server-side error occurs, it will be processed later
					 * among the other results.
					 */
					if (!PQsendPrepare(st->con, name,
									   commands[j]->argv[0], commands[j]->argc - 1, NULL))
						pg_log_error("%s", PQerrorMessage(st->con));
				}
			}
			st->prepared[st->use_file] = true;
		}

		getQueryParams(st, command, params);
		preparedStatementName(name, st->use_file, st->command);

		pg_log_debug("client %d sending %s", st->id, name);
		r = PQsendQueryPrepared(st->con, name, command->argc - 1,
								params, NULL, NULL, 0);
	}
	else						/* unknown sql mode */
		r = 0;

	if (r == 0)
	{
		pg_log_debug("client %d could not send %s", st->id, command->argv[0]);
		return false;
	}
	else
		return true;
}

/*
 * Process query response from the backend.
 *
 * If varprefix is not NULL, it's the variable name prefix where to store
 * the results of the *last* command (META_GSET) or *all* commands
 * (META_ASET).
 *
 * Returns true if everything is A-OK, false if any error occurs.
 */
static bool
readCommandResponse(CState *st, MetaCommand meta, char *varprefix)
{
	PGresult   *res;
	PGresult   *next_res;
	int			qrynum = 0;

	/*
	 * varprefix should be set only with \gset or \aset, and \endpipeline and
	 * SQL commands do not need it.
	 */
	Assert((meta == META_NONE && varprefix == NULL) ||
		   ((meta == META_ENDPIPELINE) && varprefix == NULL) ||
		   ((meta == META_GSET || meta == META_ASET) && varprefix != NULL));

	res = PQgetResult(st->con);

	while (res != NULL)
	{
		bool		is_last;

		/* peek at the next result to know whether the current is last */
		next_res = PQgetResult(st->con);
		is_last = (next_res == NULL);

		switch (PQresultStatus(res))
		{
			case PGRES_COMMAND_OK:	/* non-SELECT commands */
			case PGRES_EMPTY_QUERY: /* may be used for testing no-op overhead */
				if (is_last && meta == META_GSET)
				{
					pg_log_error("client %d script %d command %d query %d: expected one row, got %d",
								 st->id, st->use_file, st->command, qrynum, 0);
					goto error;
				}
				break;

			case PGRES_TUPLES_OK:
				if ((is_last && meta == META_GSET) || meta == META_ASET)
				{
					int			ntuples = PQntuples(res);

					if (meta == META_GSET && ntuples != 1)
					{
						/* under \gset, report the error */
						pg_log_error("client %d script %d command %d query %d: expected one row, got %d",
									 st->id, st->use_file, st->command, qrynum, PQntuples(res));
						goto error;
					}
					else if (meta == META_ASET && ntuples <= 0)
					{
						/* coldly skip empty result under \aset */
						break;
					}

					/* store results into variables */
					for (int fld = 0; fld < PQnfields(res); fld++)
					{
						char	   *varname = PQfname(res, fld);

						/* allocate varname only if necessary, freed below */
						if (*varprefix != '\0')
							varname = psprintf("%s%s", varprefix, varname);

						/* store last row result as a string */
						if (!putVariable(st, meta == META_ASET ? "aset" : "gset", varname,
										 PQgetvalue(res, ntuples - 1, fld)))
						{
							/* internal error */
							pg_log_error("client %d script %d command %d query %d: error storing into variable %s",
										 st->id, st->use_file, st->command, qrynum, varname);
							goto error;
						}

						if (*varprefix != '\0')
							pg_free(varname);
					}
				}
				/* otherwise the result is simply thrown away by PQclear below */
				break;

			case PGRES_PIPELINE_SYNC:
				pg_log_debug("client %d pipeline ending", st->id);
				if (PQexitPipelineMode(st->con) != 1)
					pg_log_error("client %d failed to exit pipeline mode: %s", st->id,
								 PQerrorMessage(st->con));
				break;

			default:
				/* anything else is unexpected */
				pg_log_error("client %d script %d aborted in command %d query %d: %s",
							 st->id, st->use_file, st->command, qrynum,
							 PQerrorMessage(st->con));
				goto error;
		}

		PQclear(res);
		qrynum++;
		res = next_res;
	}

	if (qrynum == 0)
	{
		pg_log_error("client %d command %d: no results", st->id, st->command);
		return false;
	}

	return true;

error:
	PQclear(res);
	PQclear(next_res);
	do
	{
		res = PQgetResult(st->con);
		PQclear(res);
	} while (res);

	return false;
}

/*
 * Parse the argument to a \sleep command, and return the requested amount
 * of delay, in microseconds.  Returns true on success, false on error.
 */
static bool
evaluateSleep(CState *st, int argc, char **argv, int *usecs)
{
	char	   *var;
