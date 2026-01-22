	if (has_loose_object(repl)) {
		const char *path = sha1_file_name(sha1);

		die("loose object %s (stored in %s) is corrupt",
		    sha1_to_hex(repl), path);
	}

	if ((p = has_packed_and_bad(repl)) != NULL)
		die("packed object %s (stored in %s) is corrupt",
		    sha1_to_hex(repl), p->pack_name);

	return NULL;
}

