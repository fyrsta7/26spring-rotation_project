static const char **prepare_shell_cmd(const char **argv)
{
	int argc, nargc = 0;
	const char **nargv;

	for (argc = 0; argv[argc]; argc++)
		; /* just counting */
	/* +1 for NULL, +3 for "sh -c" plus extra $0 */
	nargv = xmalloc(sizeof(*nargv) * (argc + 1 + 3));

	if (argc < 1)
		die("BUG: shell command is empty");

	nargv[nargc++] = "sh";
	nargv[nargc++] = "-c";

	if (argc < 2)
		nargv[nargc++] = argv[0];
	else {
		struct strbuf arg0 = STRBUF_INIT;
		strbuf_addf(&arg0, "%s \"$@\"", argv[0]);
		nargv[nargc++] = strbuf_detach(&arg0, NULL);
	}

	for (argc = 0; argv[argc]; argc++)
		nargv[nargc++] = argv[argc];
	nargv[nargc] = NULL;

	return nargv;
}
