		cb(negotiator, cache.items[i]);
}

static struct commit *deref_without_lazy_fetch(const struct object_id *oid,
					       int mark_tags_complete)
{
	enum object_type type;
	struct object_info info = { .typep = &type };

	while (1) {
		if (oid_object_info_extended(the_repository, oid, &info,
					     OBJECT_INFO_SKIP_FETCH_OBJECT | OBJECT_INFO_QUICK))
			return NULL;
		if (type == OBJ_TAG) {
			struct tag *tag = (struct tag *)
				parse_object(the_repository, oid);

			if (!tag->tagged)
				return NULL;
			if (mark_tags_complete)
				tag->object.flags |= COMPLETE;
			oid = &tag->tagged->oid;
		} else {
			break;
		}
	}

	if (type == OBJ_COMMIT) {
		struct commit *commit = lookup_commit(the_repository, oid);
		if (!commit || repo_parse_commit(the_repository, commit))
			return NULL;
		return commit;
	}
