			return -1;
		if (revs->topo_order)
			sort_in_topological_order(&revs->commits, revs->sort_order);
	} else if (revs->topo_order)
		init_topo_walk(revs);
	if (revs->line_level_traverse && want_ancestry(revs))
		/*
		 * At the moment we can only do line-level log with parent
		 * rewriting by performing this expensive pre-filtering step.
		 * If parent rewriting is not requested, then we rather
		 * perform the line-level log filtering during the regular
		 * history traversal.
		 */
		line_log_filter(revs);
	if (revs->simplify_merges)
		simplify_merges(revs);
	if (revs->children.name)
		set_children(revs);

	return 0;
}

static enum rewrite_result rewrite_one_1(struct rev_info *revs,
					 struct commit **pp,
					 struct prio_queue *queue)
{
	for (;;) {
		struct commit *p = *pp;
		if (!revs->limited)
			if (process_parents(revs, p, NULL, queue) < 0)
				return rewrite_one_error;
		if (p->object.flags & UNINTERESTING)
			return rewrite_one_ok;
		if (!(p->object.flags & TREESAME))
			return rewrite_one_ok;
		if (!p->parents)
			return rewrite_one_noparents;
		if (!(p = one_relevant_parent(revs, p->parents)))
			return rewrite_one_ok;
		*pp = p;
	}
}

static void merge_queue_into_list(struct prio_queue *q, struct commit_list **list)
{
	while (q->nr) {
		struct commit *item = prio_queue_peek(q);
		struct commit_list *p = *list;

		if (p && p->item->date >= item->date)
			list = &p->next;
		else {
			p = commit_list_insert(item, list);
			list = &p->next; /* skip newly added item */
			prio_queue_get(q); /* pop item */
		}
	}
}

static enum rewrite_result rewrite_one(struct rev_info *revs, struct commit **pp)
