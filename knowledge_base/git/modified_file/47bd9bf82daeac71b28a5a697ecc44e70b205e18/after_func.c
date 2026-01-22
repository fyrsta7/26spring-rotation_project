		die_errno("Unable to open marks file %s for writing.", file);

	for (i = 0; i < idnums.size; i++) {
		if (deco->base && deco->base->type == 1) {
			mark = ptr_to_mark(deco->decoration);
			if (fprintf(f, ":%"PRIu32" %s\n", mark,
				sha1_to_hex(deco->base->sha1)) < 0) {
			    e = 1;
			    break;
			}
		}
		deco++;
	}

	e |= ferror(f);
	e |= fclose(f);
	if (e)
		error("Unable to write marks file %s.", file);
}

static void import_marks(char *input_file)
{
	char line[512];
	FILE *f = fopen(input_file, "r");
	if (!f)
		die_errno("cannot read '%s'", input_file);

	while (fgets(line, sizeof(line), f)) {
		uint32_t mark;
		char *line_end, *mark_end;
		unsigned char sha1[20];
		struct object *object;
		struct commit *commit;
		enum object_type type;

		line_end = strchr(line, '\n');
		if (line[0] != ':' || !line_end)
			die("corrupt mark line: %s", line);
		*line_end = '\0';

		mark = strtoumax(line + 1, &mark_end, 10);
		if (!mark || mark_end == line + 1
			|| *mark_end != ' ' || get_sha1(mark_end + 1, sha1))
			die("corrupt mark line: %s", line);

		if (last_idnum < mark)
			last_idnum = mark;

		type = sha1_object_info(sha1, NULL);
		if (type < 0)
			die("object not found: %s", sha1_to_hex(sha1));
