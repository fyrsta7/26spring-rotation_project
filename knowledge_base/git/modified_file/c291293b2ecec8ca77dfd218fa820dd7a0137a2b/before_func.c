	for (i = 0; i < nr_sought; i++) {
		struct object_id oid;
		const char *p;

		ref = sought[i];
		if (ref->match_status != REF_NOT_MATCHED)
			continue;
		if (parse_oid_hex(ref->name, &oid, &p) ||
		    *p != '\0' ||
		    oidcmp(&oid, &ref->old_oid))
			continue;

		if ((allow_unadvertised_object_request &
		     (ALLOW_TIP_SHA1 | ALLOW_REACHABLE_SHA1)) ||
		    tip_oids_contain(&tip_oids, unmatched, newlist,
				     &ref->old_oid)) {
			ref->match_status = REF_MATCHED;
			*newtail = copy_ref(ref);
			newtail = &(*newtail)->next;
		} else {
			ref->match_status = REF_UNADVERTISED_NOT_ALLOWED;
		}
	}

	oidset_clear(&tip_oids);
	for (ref = unmatched; ref; ref = next) {
		next = ref->next;
		free(ref);
	}

	*refs = newlist;
}

static void mark_alternate_complete(struct object *obj)
{
	mark_complete(&obj->oid);
}

static int everything_local(struct fetch_pack_args *args,
			    struct ref **refs,
			    struct ref **sought, int nr_sought)
{
	struct ref *ref;
	int retval;
	timestamp_t cutoff = 0;

	save_commit_buffer = 0;

	for (ref = *refs; ref; ref = ref->next) {
		struct object *o;

		if (!has_object_file(&ref->old_oid))
			continue;

		o = parse_object(&ref->old_oid);
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

