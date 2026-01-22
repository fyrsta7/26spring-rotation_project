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

	object = parse_object(revs->repo, oid);
