 * get_major_server_version()
 *
 * gets the version (in unsigned int form) for the given datadir. Assumes
 * that datadir is an absolute path to a valid pgdata directory. The version
 * is retrieved by reading the PG_VERSION file.
 */
uint32
get_major_server_version(ClusterInfo *cluster)
{
	FILE	   *version_fd;
	char		ver_filename[MAXPGPATH];
	int			integer_version = 0;
	int			fractional_version = 0;

	snprintf(ver_filename, sizeof(ver_filename), "%s/PG_VERSION",
			 cluster->pgdata);
	if ((version_fd = fopen(ver_filename, "r")) == NULL)
		pg_log(PG_FATAL, "could not open version file: %s\n", ver_filename);

	if (fscanf(version_fd, "%63s", cluster->major_version_str) == 0 ||
		sscanf(cluster->major_version_str, "%d.%d", &integer_version,
			   &fractional_version) != 2)
		pg_log(PG_FATAL, "could not get version from %s\n", cluster->pgdata);

	fclose(version_fd);

	return (100 * integer_version + fractional_version) * 100;
}


static void
stop_postmaster_atexit(void)
{
	stop_postmaster(true);

}


void
start_postmaster(ClusterInfo *cluster)
{
	char		cmd[MAXPGPATH * 4 + 1000];
	PGconn	   *conn;
	bool		exit_hook_registered = false;
	bool		pg_ctl_return = false;
	char		socket_string[MAXPGPATH + 200];

	if (!exit_hook_registered)
	{
		atexit(stop_postmaster_atexit);
		exit_hook_registered = true;
	}

	socket_string[0] = '\0';

#ifdef HAVE_UNIX_SOCKETS
	/* prevent TCP/IP connections, restrict socket access */
	strcat(socket_string,
		   " -c listen_addresses='' -c unix_socket_permissions=0700");

	/* Have a sockdir?  Tell the postmaster. */
	if (cluster->sockdir)
		snprintf(socket_string + strlen(socket_string),
				 sizeof(socket_string) - strlen(socket_string),
				 " -c %s='%s'",
				 (GET_MAJOR_VERSION(cluster->major_version) < 903) ?
				 "unix_socket_directory" : "unix_socket_directories",
				 cluster->sockdir);
#endif

	/*
