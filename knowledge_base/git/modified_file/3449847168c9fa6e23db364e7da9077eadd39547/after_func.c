		 * to remove it:
		 */
		if (!path_copy.len)
			strbuf_addstr(&path_copy, path);

		if (!remove_dir_recursively(&path_copy, REMOVE_DIR_EMPTY_ONLY))
