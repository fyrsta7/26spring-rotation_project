		sz = send_client_data(1, data, 1);
		if (sz < 0)
			goto fail;
		fprintf(stderr, "flushed.\n");
	}
	if (use_sideband)
		packet_flush(1);
	return;

 fail:
	send_client_data(3, abort_msg, sizeof(abort_msg));
	die("git upload-pack: %s", abort_msg);
}

static int got_sha1(char *hex, unsigned char *sha1)
{
	struct object *o;
	int we_knew_they_have = 0;

	if (get_sha1_hex(hex, sha1))
		die("git upload-pack: expected SHA1 object, got '%s'", hex);
	if (!has_sha1_file(sha1))
		return -1;

	o = parse_object(sha1);
	if (!o)
		die("oops (%s)", sha1_to_hex(sha1));
	if (o->type == OBJ_COMMIT) {
		struct commit_list *parents;
		struct commit *commit = (struct commit *)o;
		if (o->flags & THEY_HAVE)
			we_knew_they_have = 1;
		else
