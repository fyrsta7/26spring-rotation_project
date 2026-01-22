			int s = sprintf(keep_arg,
					"--keep=fetch-pack %"PRIuMAX " on ", (uintmax_t) getpid());
			if (gethostname(keep_arg + s, sizeof(keep_arg) - s))
				strcpy(keep_arg + s, "localhost");
			*av++ = keep_arg;
		}
	}
	else {
		*av++ = "unpack-objects";
		if (args.quiet || args.no_progress)
			*av++ = "-q";
	}
	if (*hdr_arg)
		*av++ = hdr_arg;
	if (fetch_fsck_objects >= 0
	    ? fetch_fsck_objects
	    : transfer_fsck_objects >= 0
	    ? transfer_fsck_objects
	    : 0)
		*av++ = "--strict";
	*av++ = NULL;

	cmd.in = demux.out;
	cmd.git_cmd = 1;
	if (start_command(&cmd))
		die("fetch-pack: unable to fork off %s", argv[0]);
	if (do_keep && pack_lockfile) {
		*pack_lockfile = index_pack_lockfile(cmd.out);
		close(cmd.out);
	}

	if (finish_command(&cmd))
		die("%s failed", argv[0]);
	if (use_sideband && finish_async(&demux))
		die("error in sideband demultiplexer");
	return 0;
}

static struct ref *do_fetch_pack(int fd[2],
		const struct ref *orig_ref,
		int nr_match,
		char **match,
		char **pack_lockfile)
{
	struct ref *ref = copy_ref_list(orig_ref);
	unsigned char sha1[20];

	sort_ref_list(&ref, ref_compare_name);

	if (is_repository_shallow() && !server_supports("shallow"))
		die("Server does not support shallow clients");
	if (server_supports("multi_ack_detailed")) {
		if (args.verbose)
			fprintf(stderr, "Server supports multi_ack_detailed\n");
		multi_ack = 2;
		if (server_supports("no-done")) {
			if (args.verbose)
				fprintf(stderr, "Server supports no-done\n");
			if (args.stateless_rpc)
				no_done = 1;
		}
