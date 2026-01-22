	mark_tree_contents_uninteresting(tree);
}

void mark_parents_uninteresting(struct commit *commit)
{
	struct commit_list *parents = NULL, *l;

	for (l = commit->parents; l; l = l->next)
		commit_list_insert(l->item, &parents);

	while (parents) {
		struct commit *commit = pop_commit(&parents);

		while (commit) {
			/*
			 * A missing commit is ok iff its parent is marked
			 * uninteresting.
			 *
			 * We just mark such a thing parsed, so that when
			 * it is popped next time around, we won't be trying
			 * to parse it and get an error.
			 */
			if (!commit->object.parsed &&
			    !has_object_file(&commit->object.oid))
				commit->object.parsed = 1;

			if (commit->object.flags & UNINTERESTING)
				break;

			commit->object.flags |= UNINTERESTING;

