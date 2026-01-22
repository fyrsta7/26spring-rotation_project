	struct object_id oid;
	struct object *obj;
	if (get_oid("HEAD", &oid))
		return;
	obj = parse_object(revs->repo, &oid);
	if (!obj)
		return;
	add_pending_object(revs, obj, "HEAD");
}

static struct object *get_reference(struct rev_info *revs, const char *name,
				    const struct object_id *oid,
				    unsigned int flags)
{
	struct object *object;

	/*
	 * If the repository has commit graphs, repo_parse_commit() avoids
	 * reading the object buffer, so use it whenever possible.
	 */
	if (oid_object_info(revs->repo, oid, NULL) == OBJ_COMMIT) {
		struct commit *c = lookup_commit(revs->repo, oid);
		if (!repo_parse_commit(revs->repo, c))
			object = (struct object *) c;
		else
			object = NULL;
	} else {
