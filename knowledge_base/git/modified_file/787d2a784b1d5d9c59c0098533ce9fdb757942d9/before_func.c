		(revs->min_age == -1 || revs->min_age > date);
}

int create_bundle(struct bundle_header *header, const char *path,
		int argc, const char **argv)
{
	static struct lock_file lock;
	int bundle_fd = -1;
	int bundle_to_stdout;
	const char **argv_boundary = xmalloc((argc + 4) * sizeof(const char *));
	const char **argv_pack = xmalloc(5 * sizeof(const char *));
	int i, ref_count = 0;
	char buffer[1024];
	struct rev_info revs;
	struct child_process rls;
	FILE *rls_fout;

	bundle_to_stdout = !strcmp(path, "-");
	if (bundle_to_stdout)
		bundle_fd = 1;
	else
		bundle_fd = hold_lock_file_for_update(&lock, path,
						      LOCK_DIE_ON_ERROR);

	/* write signature */
	write_or_die(bundle_fd, bundle_signature, strlen(bundle_signature));

	/* init revs to list objects for pack-objects later */
	save_commit_buffer = 0;
	init_revisions(&revs, NULL);

	/* write prerequisites */
	memcpy(argv_boundary + 3, argv + 1, argc * sizeof(const char *));
	argv_boundary[0] = "rev-list";
	argv_boundary[1] = "--boundary";
	argv_boundary[2] = "--pretty=oneline";
	argv_boundary[argc + 2] = NULL;
	memset(&rls, 0, sizeof(rls));
	rls.argv = argv_boundary;
	rls.out = -1;
	rls.git_cmd = 1;
	if (start_command(&rls))
		return -1;
	rls_fout = xfdopen(rls.out, "r");
	while (fgets(buffer, sizeof(buffer), rls_fout)) {
		unsigned char sha1[20];
		if (buffer[0] == '-') {
			write_or_die(bundle_fd, buffer, strlen(buffer));
			if (!get_sha1_hex(buffer + 1, sha1)) {
				struct object *object = parse_object(sha1);
				object->flags |= UNINTERESTING;
				add_pending_object(&revs, object, buffer);
			}
		} else if (!get_sha1_hex(buffer, sha1)) {
			struct object *object = parse_object(sha1);
			object->flags |= SHOWN;
		}
	}
	fclose(rls_fout);
	if (finish_command(&rls))
		return error("rev-list died");

	/* write references */
	argc = setup_revisions(argc, argv, &revs, NULL);

	if (argc > 1)
		return error("unrecognized argument: %s'", argv[1]);

	object_array_remove_duplicates(&revs.pending);

	for (i = 0; i < revs.pending.nr; i++) {
		struct object_array_entry *e = revs.pending.objects + i;
		unsigned char sha1[20];
		char *ref;
		const char *display_ref;
		int flag;

		if (e->item->flags & UNINTERESTING)
			continue;
		if (dwim_ref(e->name, strlen(e->name), sha1, &ref) != 1)
			continue;
		if (!resolve_ref(e->name, sha1, 1, &flag))
			flag = 0;
		display_ref = (flag & REF_ISSYMREF) ? e->name : ref;

		if (e->item->type == OBJ_TAG &&
				!is_tag_in_date_range(e->item, &revs)) {
			e->item->flags |= UNINTERESTING;
			continue;
		}

		/*
		 * Make sure the refs we wrote out is correct; --max-count and
		 * other limiting options could have prevented all the tips
		 * from getting output.
		 *
		 * Non commit objects such as tags and blobs do not have
		 * this issue as they are not affected by those extra
		 * constraints.
		 */
		if (!(e->item->flags & SHOWN) && e->item->type == OBJ_COMMIT) {
			warning("ref '%s' is excluded by the rev-list options",
				e->name);
			free(ref);
			continue;
		}
		/*
		 * If you run "git bundle create bndl v1.0..v2.0", the
		 * name of the positive ref is "v2.0" but that is the
		 * commit that is referenced by the tag, and not the tag
		 * itself.
		 */
		if (hashcmp(sha1, e->item->sha1)) {
			/*
			 * Is this the positive end of a range expressed
			 * in terms of a tag (e.g. v2.0 from the range
			 * "v1.0..v2.0")?
			 */
			struct commit *one = lookup_commit_reference(sha1);
			struct object *obj;

			if (e->item == &(one->object)) {
				/*
				 * Need to include e->name as an
				 * independent ref to the pack-objects
				 * input, so that the tag is included
				 * in the output; otherwise we would
				 * end up triggering "empty bundle"
				 * error.
				 */
				obj = parse_object(sha1);
				obj->flags |= SHOWN;
				add_pending_object(&revs, obj, e->name);
			}
			free(ref);
			continue;
		}

		ref_count++;
		write_or_die(bundle_fd, sha1_to_hex(e->item->sha1), 40);
		write_or_die(bundle_fd, " ", 1);
		write_or_die(bundle_fd, display_ref, strlen(display_ref));
		write_or_die(bundle_fd, "\n", 1);
		free(ref);
	}
	if (!ref_count)
		die ("Refusing to create empty bundle.");

	/* end header */
	write_or_die(bundle_fd, "\n", 1);

	/* write pack */
	argv_pack[0] = "pack-objects";
	argv_pack[1] = "--all-progress-implied";
	argv_pack[2] = "--stdout";
	argv_pack[3] = "--thin";
	argv_pack[4] = NULL;
