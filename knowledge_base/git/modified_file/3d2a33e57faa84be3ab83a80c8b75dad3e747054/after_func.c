	if (args.stateless_rpc) {
		conn = NULL;
		fd[0] = 0;
		fd[1] = 1;
	} else {
		conn = git_connect(fd, (char *)dest, args.uploadpack,
				   args.verbose ? CONNECT_VERBOSE : 0);
	}

	get_remote_heads(fd[0], &ref, 0, NULL);

	ref = fetch_pack(&args, fd, conn, ref, dest,
		nr_heads, heads, pack_lockfile_ptr);
	if (pack_lockfile) {
		printf("lock %s\n", pack_lockfile);
		fflush(stdout);
	}
	close(fd[0]);
	close(fd[1]);
	if (finish_connect(conn))
		ref = NULL;
	ret = !ref;

	if (!ret && nr_heads) {
		/* If the heads to pull were given, we should have
		 * consumed all of them by matching the remote.
		 * Otherwise, 'git fetch remote no-such-ref' would
		 * silently succeed without issuing an error.
		 */
		for (i = 0; i < nr_heads; i++)
			if (heads[i] && heads[i][0]) {
				error("no such remote ref %s", heads[i]);
				ret = 1;
			}
	}
	while (ref) {
		printf("%s %s\n",
		       sha1_to_hex(ref->old_sha1), ref->name);
		ref = ref->next;
	}

	return ret;
}

static int compare_heads(const void *a, const void *b)
{
	return strcmp(*(const char **)a, *(const char **)b);
}

struct ref *fetch_pack(struct fetch_pack_args *my_args,
		       int fd[], struct child_process *conn,
		       const struct ref *ref,
		const char *dest,
		int nr_heads,
		char **heads,
		char **pack_lockfile)
{
	struct stat st;
	struct ref *ref_cpy;

	fetch_pack_setup();
	if (&args != my_args)
		memcpy(&args, my_args, sizeof(args));
