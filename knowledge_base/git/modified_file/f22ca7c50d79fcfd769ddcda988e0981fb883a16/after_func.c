static struct object *parse_object_cheap(const unsigned char *sha1)
{
	struct object *o;

	if ((o = parse_object(sha1)) == NULL)
		return NULL;
	if (o->type == commit_type) {
		struct commit *commit = (struct commit *)o;
		free(commit->buffer);
		commit->buffer = NULL;
	} else if (o->type == tree_type) {
		struct tree *tree = (struct tree *)o;
		struct tree_entry_list *e, *n;
		for (e = tree->entries; e; e = n) {
			free(e->name);
			e->name = NULL;
			n = e->next;
			free(e);
		}
		tree->entries = NULL;
	}
	return o;
}
