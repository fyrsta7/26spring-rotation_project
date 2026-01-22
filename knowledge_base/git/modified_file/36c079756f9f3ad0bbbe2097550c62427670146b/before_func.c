		*cache = new_entry;
}

static int add_parents_to_list(struct rev_info *revs, struct commit *commit,
		    struct commit_list **list, struct commit_list **cache_ptr)
{
	struct commit_list *parent = commit->parents;
	unsigned left_flag;
	struct commit_list *cached_base = cache_ptr ? *cache_ptr : NULL;

	if (commit->object.flags & ADDED)
		return 0;
	commit->object.flags |= ADDED;

	/*
	 * If the commit is uninteresting, don't try to
	 * prune parents - we want the maximal uninteresting
	 * set.
	 *
	 * Normally we haven't parsed the parent
	 * yet, so we won't have a parent of a parent
	 * here. However, it may turn out that we've
	 * reached this commit some other way (where it
	 * wasn't uninteresting), in which case we need
	 * to mark its parents recursively too..
	 */
	if (commit->object.flags & UNINTERESTING) {
		while (parent) {
			struct commit *p = parent->item;
			parent = parent->next;
			if (parse_commit(p) < 0)
				return -1;
			p->object.flags |= UNINTERESTING;
			if (p->parents)
				mark_parents_uninteresting(p);
			if (p->object.flags & SEEN)
				continue;
			p->object.flags |= SEEN;
			insert_by_date_cached(p, list, cached_base, cache_ptr);
		}
		return 0;
	}

	/*
	 * Ok, the commit wasn't uninteresting. Try to
	 * simplify the commit history and find the parent
	 * that has no differences in the path set if one exists.
	 */
	try_to_simplify_commit(revs, commit);

	if (revs->no_walk)
		return 0;

	left_flag = (commit->object.flags & SYMMETRIC_LEFT);

	for (parent = commit->parents; parent; parent = parent->next) {
		struct commit *p = parent->item;

		if (parse_commit(p) < 0)
			return -1;
		p->object.flags |= left_flag;
		if (!(p->object.flags & SEEN)) {
			p->object.flags |= SEEN;
			insert_by_date_cached(p, list, cached_base, cache_ptr);
		}
		if(revs->first_parent_only)
			break;
	}
	return 0;
}

static void cherry_pick_list(struct commit_list *list, struct rev_info *revs)
{
	struct commit_list *p;
	int left_count = 0, right_count = 0;
