	bool		scale_given = false;

	bool		benchmarking_option_set = false;
	bool		initialization_option_set = false;
	bool		internal_script_used = false;

	CState	   *state;			/* status of clients */
	TState	   *threads;		/* array of thread */

	pg_time_usec_t
				start_time,		/* start up time */
				bench_start = 0,	/* first recorded benchmarking time */
				conn_total_duration;	/* cumulated connection time in
										 * threads */
	int64		latency_late = 0;
	StatsData	stats;
	int			weight;

	int			i;
	int			nclients_dealt;

#ifdef HAVE_GETRLIMIT
	struct rlimit rlim;
#endif

	PGconn	   *con;
	char	   *env;

	int			exit_code = 0;
	struct timeval tv;

	/*
	 * Record difference between Unix time and instr_time time.  We'll use
	 * this for logging and aggregation.
	 */
	gettimeofday(&tv, NULL);
	epoch_shift = tv.tv_sec * INT64CONST(1000000) + tv.tv_usec - pg_time_now();

	pg_logging_init(argv[0]);
	progname = get_progname(argv[0]);

	if (argc > 1)
	{
		if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-?") == 0)
		{
			usage();
			exit(0);
		}
		if (strcmp(argv[1], "--version") == 0 || strcmp(argv[1], "-V") == 0)
		{
			puts("pgbench (PostgreSQL) " PG_VERSION);
			exit(0);
		}
	}

	state = (CState *) pg_malloc0(sizeof(CState));

	/* set random seed early, because it may be used while parsing scripts. */
	if (!set_random_seed(getenv("PGBENCH_RANDOM_SEED")))
	{
		pg_log_fatal("error while setting random seed from PGBENCH_RANDOM_SEED environment variable");
		exit(1);
	}

	while ((c = getopt_long(argc, argv, "iI:h:nvp:dqb:SNc:j:Crs:t:T:U:lf:D:F:M:P:R:L:", long_options, &optindex)) != -1)
	{
		char	   *script;

		switch (c)
		{
			case 'i':
				is_init_mode = true;
				break;
			case 'I':
				if (initialize_steps)
					pg_free(initialize_steps);
				initialize_steps = pg_strdup(optarg);
				checkInitSteps(initialize_steps);
				initialization_option_set = true;
				break;
			case 'h':
				pghost = pg_strdup(optarg);
				break;
			case 'n':
				is_no_vacuum = true;
				break;
			case 'v':
				benchmarking_option_set = true;
				do_vacuum_accounts = true;
				break;
			case 'p':
				pgport = pg_strdup(optarg);
				break;
			case 'd':
				pg_logging_increase_verbosity();
				break;
			case 'c':
				benchmarking_option_set = true;
				if (!option_parse_int(optarg, "-c/--clients", 1, INT_MAX,
									  &nclients))
				{
					exit(1);
				}
#ifdef HAVE_GETRLIMIT
#ifdef RLIMIT_NOFILE			/* most platforms use RLIMIT_NOFILE */
				if (getrlimit(RLIMIT_NOFILE, &rlim) == -1)
#else							/* but BSD doesn't ... */
				if (getrlimit(RLIMIT_OFILE, &rlim) == -1)
#endif							/* RLIMIT_NOFILE */
				{
					pg_log_fatal("getrlimit failed: %m");
					exit(1);
				}
				if (rlim.rlim_cur < nclients + 3)
				{
					pg_log_fatal("need at least %d open files, but system limit is %ld",
								 nclients + 3, (long) rlim.rlim_cur);
					pg_log_info("Reduce number of clients, or use limit/ulimit to increase the system limit.");
					exit(1);
				}
#endif							/* HAVE_GETRLIMIT */
				break;
			case 'j':			/* jobs */
				benchmarking_option_set = true;
				if (!option_parse_int(optarg, "-j/--jobs", 1, INT_MAX,
									  &nthreads))
				{
					exit(1);
				}
#ifndef ENABLE_THREAD_SAFETY
				if (nthreads != 1)
				{
					pg_log_fatal("threads are not supported on this platform; use -j1");
					exit(1);
				}
#endif							/* !ENABLE_THREAD_SAFETY */
				break;
			case 'C':
				benchmarking_option_set = true;
				is_connect = true;
				break;
			case 'r':
				benchmarking_option_set = true;
				report_per_command = true;
				break;
			case 's':
				scale_given = true;
				if (!option_parse_int(optarg, "-s/--scale", 1, INT_MAX,
									  &scale))
					exit(1);
				break;
			case 't':
				benchmarking_option_set = true;
				if (!option_parse_int(optarg, "-t/--transactions", 1, INT_MAX,
									  &nxacts))
					exit(1);
				break;
			case 'T':
				benchmarking_option_set = true;
				if (!option_parse_int(optarg, "-T/--time", 1, INT_MAX,
									  &duration))
					exit(1);
				break;
			case 'U':
				username = pg_strdup(optarg);
				break;
			case 'l':
				benchmarking_option_set = true;
				use_log = true;
				break;
			case 'q':
				initialization_option_set = true;
				use_quiet = true;
				break;
			case 'b':
				if (strcmp(optarg, "list") == 0)
				{
					listAvailableScripts();
					exit(0);
				}
				weight = parseScriptWeight(optarg, &script);
				process_builtin(findBuiltin(script), weight);
				benchmarking_option_set = true;
				internal_script_used = true;
				break;
			case 'S':
				process_builtin(findBuiltin("select-only"), 1);
				benchmarking_option_set = true;
				internal_script_used = true;
				break;
			case 'N':
				process_builtin(findBuiltin("simple-update"), 1);
				benchmarking_option_set = true;
				internal_script_used = true;
				break;
			case 'f':
				weight = parseScriptWeight(optarg, &script);
				process_file(script, weight);
				benchmarking_option_set = true;
				break;
			case 'D':
				{
					char	   *p;

					benchmarking_option_set = true;

					if ((p = strchr(optarg, '=')) == NULL || p == optarg || *(p + 1) == '\0')
					{
						pg_log_fatal("invalid variable definition: \"%s\"", optarg);
						exit(1);
					}

					*p++ = '\0';
					if (!putVariable(&state[0], "option", optarg, p))
						exit(1);
				}
				break;
			case 'F':
				initialization_option_set = true;
				if (!option_parse_int(optarg, "-F/--fillfactor", 10, 100,
									  &fillfactor))
					exit(1);
				break;
			case 'M':
				benchmarking_option_set = true;
				for (querymode = 0; querymode < NUM_QUERYMODE; querymode++)
					if (strcmp(optarg, QUERYMODE[querymode]) == 0)
						break;
				if (querymode >= NUM_QUERYMODE)
				{
					pg_log_fatal("invalid query mode (-M): \"%s\"", optarg);
					exit(1);
				}
				break;
			case 'P':
				benchmarking_option_set = true;
				if (!option_parse_int(optarg, "-P/--progress", 1, INT_MAX,
									  &progress))
					exit(1);
				break;
			case 'R':
				{
					/* get a double from the beginning of option value */
					double		throttle_value = atof(optarg);

					benchmarking_option_set = true;

					if (throttle_value <= 0.0)
					{
						pg_log_fatal("invalid rate limit: \"%s\"", optarg);
						exit(1);
					}
					/* Invert rate limit into per-transaction delay in usec */
					throttle_delay = 1000000.0 / throttle_value;
				}
				break;
