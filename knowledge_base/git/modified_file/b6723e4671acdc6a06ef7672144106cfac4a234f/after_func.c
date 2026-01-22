		case CONTAINS_UNKNOWN:
			push_to_contains_stack(parents->item, &contains_stack);
			break;
		}
	}
	free(contains_stack.contains_stack);
	return contains_test(candidate, want, cache, cutoff);
}

int commit_contains(struct ref_filter *filter, struct commit *commit,
		    struct commit_list *list, struct contains_cache *cache)
{
	if (filter->with_commit_tag_algo)
		return contains_tag_algo(commit, list, cache) == CONTAINS_YES;
	return is_descendant_of(commit, list);
}

static int compare_commits_by_gen(const void *_a, const void *_b)
{
	const struct commit *a = (const struct commit *)_a;
	const struct commit *b = (const struct commit *)_b;

	if (a->generation < b->generation)
		return -1;
	if (a->generation > b->generation)
		return 1;
	return 0;
}

int can_all_from_reach_with_flag(struct object_array *from,
				 unsigned int with_flag,
				 unsigned int assign_flag,
				 time_t min_commit_date,
				 uint32_t min_generation)
{
	struct commit **list = NULL;
	int i;
	int nr_commits;
	int result = 1;

	ALLOC_ARRAY(list, from->nr);
	nr_commits = 0;
	for (i = 0; i < from->nr; i++) {
		struct object *from_one = from->objects[i].item;

		if (!from_one || from_one->flags & assign_flag)
			continue;

		from_one = deref_tag(the_repository, from_one,
				     "a from object", 0);
		if (!from_one || from_one->type != OBJ_COMMIT) {
			/*
			 * no way to tell if this is reachable by
			 * looking at the ancestry chain alone, so
			 * leave a note to ourselves not to worry about
			 * this object anymore.
			 */
			from->objects[i].item->flags |= assign_flag;
			continue;
		}

		list[nr_commits] = (struct commit *)from_one;
		if (parse_commit(list[nr_commits]) ||
		    list[nr_commits]->generation < min_generation) {
			result = 0;
			goto cleanup;
		}

		nr_commits++;
	}

	QSORT(list, nr_commits, compare_commits_by_gen);

	for (i = 0; i < nr_commits; i++) {
		/* DFS from list[i] */
		struct commit_list *stack = NULL;

		list[i]->object.flags |= assign_flag;
		commit_list_insert(list[i], &stack);

		while (stack) {
			struct commit_list *parent;

			if (stack->item->object.flags & (with_flag | RESULT)) {
				pop_commit(&stack);
				if (stack)
					stack->item->object.flags |= RESULT;
				continue;
			}

			for (parent = stack->item->parents; parent; parent = parent->next) {
				if (parent->item->object.flags & (with_flag | RESULT))
