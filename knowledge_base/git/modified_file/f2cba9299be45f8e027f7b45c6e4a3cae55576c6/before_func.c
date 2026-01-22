	die("git fetch_pack: expected ACK/NAK, got '%s'", line);
}

static void send_request(int fd, struct strbuf *buf)
{
	if (args.stateless_rpc) {
		send_sideband(fd, -1, buf->buf, buf->len, LARGE_PACKET_MAX);
		packet_flush(fd);
	} else
		safe_write(fd, buf->buf, buf->len);
}

static int find_common(int fd[2], unsigned char *result_sha1,
		       struct ref *refs)
{
	int fetching;
	int count = 0, flushes = 0, retval;
	const unsigned char *sha1;
	unsigned in_vain = 0;
	int got_continue = 0;
	struct strbuf req_buf = STRBUF_INIT;
	size_t state_len = 0;

	if (args.stateless_rpc && multi_ack == 1)
		die("--stateless-rpc requires multi_ack_detailed");
	if (marked)
		for_each_ref(clear_marks, NULL);
	marked = 1;

	for_each_ref(rev_list_insert_ref, NULL);

	fetching = 0;
	for ( ; refs ; refs = refs->next) {
		unsigned char *remote = refs->old_sha1;
		const char *remote_hex;
		struct object *o;

		/*
		 * If that object is complete (i.e. it is an ancestor of a
		 * local ref), we tell them we have it but do not have to
		 * tell them about its ancestors, which they already know
		 * about.
		 *
		 * We use lookup_object here because we are only
		 * interested in the case we *know* the object is
		 * reachable and we have already scanned it.
		 */
		if (((o = lookup_object(remote)) != NULL) &&
				(o->flags & COMPLETE)) {
			continue;
		}

		remote_hex = sha1_to_hex(remote);
		if (!fetching) {
			struct strbuf c = STRBUF_INIT;
			if (multi_ack == 2)     strbuf_addstr(&c, " multi_ack_detailed");
			if (multi_ack == 1)     strbuf_addstr(&c, " multi_ack");
			if (use_sideband == 2)  strbuf_addstr(&c, " side-band-64k");
			if (use_sideband == 1)  strbuf_addstr(&c, " side-band");
			if (args.use_thin_pack) strbuf_addstr(&c, " thin-pack");
			if (args.no_progress)   strbuf_addstr(&c, " no-progress");
			if (args.include_tag)   strbuf_addstr(&c, " include-tag");
			if (prefer_ofs_delta)   strbuf_addstr(&c, " ofs-delta");
			packet_buf_write(&req_buf, "want %s%s\n", remote_hex, c.buf);
			strbuf_release(&c);
		} else
			packet_buf_write(&req_buf, "want %s\n", remote_hex);
		fetching++;
	}

	if (!fetching) {
		strbuf_release(&req_buf);
		packet_flush(fd[1]);
		return 1;
	}

	if (is_repository_shallow())
		write_shallow_commits(&req_buf, 1);
	if (args.depth > 0)
		packet_buf_write(&req_buf, "deepen %d", args.depth);
	packet_buf_flush(&req_buf);
	state_len = req_buf.len;

	if (args.depth > 0) {
		char line[1024];
		unsigned char sha1[20];

		send_request(fd[1], &req_buf);
		while (packet_read_line(fd[0], line, sizeof(line))) {
			if (!prefixcmp(line, "shallow ")) {
				if (get_sha1_hex(line + 8, sha1))
					die("invalid shallow line: %s", line);
				register_shallow(sha1);
				continue;
			}
			if (!prefixcmp(line, "unshallow ")) {
				if (get_sha1_hex(line + 10, sha1))
					die("invalid unshallow line: %s", line);
				if (!lookup_object(sha1))
					die("object not found: %s", line);
				/* make sure that it is parsed as shallow */
				if (!parse_object(sha1))
					die("error in object: %s", line);
				if (unregister_shallow(sha1))
					die("no shallow found: %s", line);
				continue;
			}
			die("expected shallow/unshallow, got %s", line);
		}
	} else if (!args.stateless_rpc)
		send_request(fd[1], &req_buf);

	if (!args.stateless_rpc) {
		/* If we aren't using the stateless-rpc interface
		 * we don't need to retain the headers.
		 */
		strbuf_setlen(&req_buf, 0);
		state_len = 0;
	}

	flushes = 0;
	retval = -1;
	while ((sha1 = get_rev())) {
		packet_buf_write(&req_buf, "have %s\n", sha1_to_hex(sha1));
		if (args.verbose)
			fprintf(stderr, "have %s\n", sha1_to_hex(sha1));
		in_vain++;
		if (!(31 & ++count)) {
			int ack;

			packet_buf_flush(&req_buf);
			send_request(fd[1], &req_buf);
			strbuf_setlen(&req_buf, state_len);
			flushes++;

			/*
			 * We keep one window "ahead" of the other side, and
			 * will wait for an ACK only on the next one
			 */
			if (!args.stateless_rpc && count == 32)
				continue;

			consume_shallow_list(fd[0]);
			do {
				ack = get_ack(fd[0], result_sha1);
				if (args.verbose && ack)
					fprintf(stderr, "got ack %d %s\n", ack,
							sha1_to_hex(result_sha1));
				switch (ack) {
				case ACK:
					flushes = 0;
					multi_ack = 0;
					retval = 0;
					goto done;
				case ACK_common:
				case ACK_ready:
				case ACK_continue: {
					struct commit *commit =
						lookup_commit(result_sha1);
					if (args.stateless_rpc
					 && ack == ACK_common
					 && !(commit->object.flags & COMMON)) {
						/* We need to replay the have for this object
						 * on the next RPC request so the peer knows
						 * it is in common with us.
						 */
						const char *hex = sha1_to_hex(result_sha1);
						packet_buf_write(&req_buf, "have %s\n", hex);
						state_len = req_buf.len;
					}
					mark_common(commit, 0, 1);
					retval = 0;
					in_vain = 0;
					got_continue = 1;
					break;
					}
				}
			} while (ack);
			flushes--;
			if (got_continue && MAX_IN_VAIN < in_vain) {
				if (args.verbose)
					fprintf(stderr, "giving up\n");
				break; /* give up */
			}
		}
