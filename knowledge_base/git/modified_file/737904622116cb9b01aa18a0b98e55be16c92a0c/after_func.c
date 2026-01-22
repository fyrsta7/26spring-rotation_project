			strbuf_addstr(dst, "tags/");
		strbuf_addstr(dst, n->tag->tag);
	} else {
		strbuf_addstr(dst, n->path);
	}
}

static void append_suffix(int depth, const struct object_id *oid, struct strbuf *dst)
{
	strbuf_addf(dst, "-%d-g%s", depth,
		    repo_find_unique_abbrev(the_repository, oid, abbrev));
}

static void describe_commit(struct object_id *oid, struct strbuf *dst)
{
	struct commit *cmit, *gave_up_on = NULL;
	struct commit_list *list;
	struct commit_name *n;
	struct possible_tag all_matches[MAX_TAGS];
	unsigned int match_cnt = 0, annotated_cnt = 0, cur_match;
	unsigned long seen_commits = 0;
	unsigned int unannotated_cnt = 0;

	cmit = lookup_commit_reference(the_repository, oid);

	n = find_commit_name(&cmit->object.oid);
	if (n && (tags || all || n->prio == 2)) {
		/*
		 * Exact match to an existing ref.
		 */
		append_name(n, dst);
		if (n->misnamed || longformat)
			append_suffix(0, n->tag ? get_tagged_oid(n->tag) : oid, dst);
		if (suffix)
			strbuf_addstr(dst, suffix);
		return;
	}

	if (!max_candidates)
		die(_("no tag exactly matches '%s'"), oid_to_hex(&cmit->object.oid));
	if (debug)
		fprintf(stderr, _("No exact match on refs or tags, searching to describe\n"));

	if (!have_util) {
		struct hashmap_iter iter;
		struct commit *c;
		struct commit_name *n;

		init_commit_names(&commit_names);
		hashmap_for_each_entry(&names, &iter, n,
					entry /* member name */) {
			c = lookup_commit_reference_gently(the_repository,
							   &n->peeled, 1);
			if (c)
				*commit_names_at(&commit_names, c) = n;
		}
		have_util = 1;
	}

	list = NULL;
	cmit->object.flags = SEEN;
	commit_list_insert(cmit, &list);
	while (list) {
		struct commit *c = pop_commit(&list);
		struct commit_list *parents = c->parents;
		struct commit_name **slot;

		seen_commits++;

		if (match_cnt == max_candidates) {
			gave_up_on = c;
			break;
		}

		slot = commit_names_peek(&commit_names, c);
		n = slot ? *slot : NULL;
		if (n) {
			if (!tags && !all && n->prio < 2) {
				unannotated_cnt++;
			} else if (match_cnt < max_candidates) {
				struct possible_tag *t = &all_matches[match_cnt++];
				t->name = n;
				t->depth = seen_commits - 1;
				t->flag_within = 1u << match_cnt;
				t->found_order = match_cnt;
				c->object.flags |= t->flag_within;
				if (n->prio == 2)
					annotated_cnt++;
			}
		}
		for (cur_match = 0; cur_match < match_cnt; cur_match++) {
			struct possible_tag *t = &all_matches[cur_match];
			if (!(c->object.flags & t->flag_within))
				t->depth++;
		}
		/* Stop if last remaining path already covered by best candidate(s) */
		if (annotated_cnt && !list) {
			int best_depth = INT_MAX;
			unsigned best_within = 0;
			for (cur_match = 0; cur_match < match_cnt; cur_match++) {
				struct possible_tag *t = &all_matches[cur_match];
				if (t->depth < best_depth) {
					best_depth = t->depth;
					best_within = t->flag_within;
				} else if (t->depth == best_depth) {
					best_within |= t->flag_within;
				}
			}
			if ((c->object.flags & best_within) == best_within) {
				if (debug)
					fprintf(stderr, _("finished search at %s\n"),
						oid_to_hex(&c->object.oid));
				break;
			}
		}
		while (parents) {
			struct commit *p = parents->item;
			repo_parse_commit(the_repository, p);
			if (!(p->object.flags & SEEN))
				commit_list_insert_by_date(p, &list);
			p->object.flags |= c->object.flags;
			parents = parents->next;

			if (first_parent)
				break;
		}
	}

	if (!match_cnt) {
		struct object_id *cmit_oid = &cmit->object.oid;
		if (always) {
			strbuf_add_unique_abbrev(dst, cmit_oid, abbrev);
			if (suffix)
				strbuf_addstr(dst, suffix);
			return;
		}
		if (unannotated_cnt)
			die(_("No annotated tags can describe '%s'.\n"
			    "However, there were unannotated tags: try --tags."),
			    oid_to_hex(cmit_oid));
		else
			die(_("No tags can describe '%s'.\n"
			    "Try --always, or create some tags."),
			    oid_to_hex(cmit_oid));
	}

	QSORT(all_matches, match_cnt, compare_pt);

	if (gave_up_on) {
		commit_list_insert_by_date(gave_up_on, &list);
		seen_commits--;
	}
	seen_commits += finish_depth_computation(&list, &all_matches[0]);
	free_commit_list(list);

	if (debug) {
		static int label_width = -1;
		if (label_width < 0) {
			int i, w;
			for (i = 0; i < ARRAY_SIZE(prio_names); i++) {
				w = strlen(_(prio_names[i]));
				if (label_width < w)
					label_width = w;
			}
		}
		for (cur_match = 0; cur_match < match_cnt; cur_match++) {
			struct possible_tag *t = &all_matches[cur_match];
			fprintf(stderr, " %-*s %8d %s\n",
				label_width, _(prio_names[t->name->prio]),
				t->depth, t->name->path);
		}
