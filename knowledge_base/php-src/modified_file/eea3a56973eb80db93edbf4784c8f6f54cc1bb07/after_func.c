	
	free(old_state);
	
	efree(free_path);
#if VIRTUAL_CWD_DEBUG
	fprintf (stderr, "virtual_file_ex() = %s\n",state->cwd);
#endif
	return (ret);
}

CWD_API int virtual_chdir(const char *path)
{
	CWDLS_FETCH();

	return virtual_file_ex(&CWDG(cwd), path, php_is_dir_ok)?-1:0;
}

CWD_API int virtual_chdir_file(const char *path, int (*p_chdir)(const char *path))
{
	int length = strlen(path);
	char *temp;
	int retval;

	if (length == 0) {
		return 1; /* Can't cd to empty string */
	}	
	while(--length >= 0 && !IS_SLASH(path[length])) {
	}

