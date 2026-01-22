static int diff_tree_stdin(char *line)
{
	int len = strlen(line);
	unsigned char sha1[20];
	struct object *obj;

	if (!len || line[len-1] != '\n')
		return -1;
	line[len-1] = 0;
	if (get_sha1_hex(line, sha1))
		return -1;
	obj = parse_object(sha1);
	if (!obj)
		return -1;
	if (obj->type == OBJ_COMMIT)
		return stdin_diff_commit((struct commit *)obj, line, len);
	if (obj->type == OBJ_TREE)
		return stdin_diff_trees((struct tree *)obj, line, len);
	error("Object %s is a %s, not a commit or tree",
	      sha1_to_hex(sha1), typename(obj->type));
	return -1;
}
