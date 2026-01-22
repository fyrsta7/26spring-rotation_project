	return 1;
}

static int add_ref_decoration(const char *refname, const struct object_id *oid,
			      int flags, void *cb_data)
{
	struct object *obj;
	enum object_type objtype;
	enum decoration_type type = DECORATION_NONE;
	struct decoration_filter *filter = (struct decoration_filter *)cb_data;

	if (filter && !ref_filter_match(refname, filter))
		return 0;

	if (starts_with(refname, git_replace_ref_base)) {
		struct object_id original_oid;
		if (!read_replace_refs)
			return 0;
		if (get_oid_hex(refname + strlen(git_replace_ref_base),
				&original_oid)) {
			warning("invalid replace ref %s", refname);
			return 0;
		}
		obj = parse_object(the_repository, &original_oid);
		if (obj)
			add_name_decoration(DECORATION_GRAFTED, "replaced", obj);
		return 0;
	}

	objtype = oid_object_info(the_repository, oid, NULL);
	if (objtype < 0)
		return 0;
	obj = lookup_object_by_type(the_repository, oid, objtype);

	if (starts_with(refname, "refs/heads/"))
		type = DECORATION_REF_LOCAL;
	else if (starts_with(refname, "refs/remotes/"))
		type = DECORATION_REF_REMOTE;
	else if (starts_with(refname, "refs/tags/"))
		type = DECORATION_REF_TAG;
	else if (!strcmp(refname, "refs/stash"))
		type = DECORATION_REF_STASH;
	else if (!strcmp(refname, "HEAD"))
		type = DECORATION_REF_HEAD;

	add_name_decoration(type, refname, obj);
	while (obj->type == OBJ_TAG) {
		obj = ((struct tag *)obj)->tagged;
		if (!obj)
			break;
		if (!obj->parsed)
			parse_object(the_repository, &obj->oid);
		add_name_decoration(DECORATION_REF_TAG, refname, obj);
