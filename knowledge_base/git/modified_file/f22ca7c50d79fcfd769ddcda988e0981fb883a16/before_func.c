static struct object *parse_object_cheap(const unsigned char *sha1)
{
	struct object *o;

	if ((o = parse_object(sha1)) == NULL)
		return NULL;
	if (o->type == commit_type) {
		struct commit *commit = (struct commit *)o;
		free(commit->buffer);
		commit->buffer = NULL;
	}
	return o;
}
