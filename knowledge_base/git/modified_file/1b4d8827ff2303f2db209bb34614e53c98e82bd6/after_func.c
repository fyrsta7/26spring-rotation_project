		die("bad object %s", name);
	}
	object->flags |= flags;
	return object;
}

void add_pending_oid(struct rev_info *revs, const char *name,
		      const struct object_id *oid, unsigned int flags)
{
	struct object *object = get_reference(revs, name, oid, flags);
	add_pending_object(revs, object, name);
}

static struct commit *handle_commit(struct rev_info *revs,
				    struct object_array_entry *entry)
{
	struct object *object = entry->item;
	const char *name = entry->name;
	const char *path = entry->path;
	unsigned int mode = entry->mode;
	unsigned long flags = object->flags;

	/*
	 * Tag object? Look what it points to..
	 */
	while (object->type == OBJ_TAG) {
		struct tag *tag = (struct tag *) object;
		if (revs->tag_objects && !(flags & UNINTERESTING))
			add_pending_object(revs, object, tag->tag);
		if (!tag->tagged)
			die("bad tag");
		object = parse_object(revs->repo, &tag->tagged->oid);
		if (!object) {
			if (revs->ignore_missing_links || (flags & UNINTERESTING))
				return NULL;
			if (revs->exclude_promisor_objects &&
			    is_promisor_object(&tag->tagged->oid))
				return NULL;
			die("bad object %s", oid_to_hex(&tag->tagged->oid));
		}
		object->flags |= flags;
		/*
		 * We'll handle the tagged object by looping or dropping
		 * through to the non-tag handlers below. Do not
		 * propagate path data from the tag's pending entry.
		 */
		path = NULL;
		mode = 0;
	}

	/*
	 * Commit object? Just return it, we'll do all the complex
	 * reachability crud.
	 */
	if (object->type == OBJ_COMMIT) {
		struct commit *commit = (struct commit *)object;

		if (parse_commit(commit) < 0)
			die("unable to parse commit %s", name);
		if (flags & UNINTERESTING) {
			mark_parents_uninteresting(commit);

			if (!revs->topo_order || !generation_numbers_enabled(the_repository))
				revs->limited = 1;
		}
		if (revs->sources) {
			char **slot = revision_sources_at(revs->sources, commit);

			if (!*slot)
				*slot = xstrdup(name);
		}
		return commit;
	}

	/*
	 * Tree object? Either mark it uninteresting, or add it
