					match_pos++;
			}
			if (!cmp)
				continue; /* we will link it later */
		}
		free(ref);
	}

	if (!args.fetch_all) {
		int i;
		for (i = 0; i < nr_match; i++) {
			ref = return_refs[i];
			if (ref) {
				*newtail = ref;
				ref->next = NULL;
				newtail = &ref->next;
			}
		}
		if (return_refs != fastarray)
			free(return_refs);
	}
	*refs = newlist;
}

static void mark_alternate_complete(const struct ref *ref, void *unused)
{
	mark_complete(NULL, ref->old_sha1, 0, NULL);
}

static int everything_local(struct ref **refs, int nr_match, char **match)
{
	struct ref *ref;
	int retval;
	unsigned long cutoff = 0;

	save_commit_buffer = 0;

	for (ref = *refs; ref; ref = ref->next) {
		struct object *o;

		o = parse_object(ref->old_sha1);
		if (!o)
			continue;

		/* We already have it -- which may mean that we were
		 * in sync with the other side at some time after
		 * that (it is OK if we guess wrong here).
		 */
		if (o->type == OBJ_COMMIT) {
			struct commit *commit = (struct commit *)o;
			if (!cutoff || cutoff < commit->date)
				cutoff = commit->date;
		}
	}

	if (!args.depth) {
		for_each_ref(mark_complete, NULL);
		for_each_alternate_ref(mark_alternate_complete, NULL);
		if (cutoff)
			mark_recent_complete_commits(cutoff);
	}

	/*
	 * Mark all complete remote refs as common refs.
	 * Don't mark them common yet; the server has to be told so first.
	 */
	for (ref = *refs; ref; ref = ref->next) {
		struct object *o = deref_tag(lookup_object(ref->old_sha1),
					     NULL, 0);

		if (!o || o->type != OBJ_COMMIT || !(o->flags & COMPLETE))
			continue;
