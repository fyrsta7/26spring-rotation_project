	if (sort_order == REV_SORT_BY_AUTHOR_DATE)
		clear_author_date_slab(&author_date);
}

/* merge-base stuff */

/* Remember to update object flag allocation in object.h */
#define PARENT1		(1u<<16)
#define PARENT2		(1u<<17)
#define STALE		(1u<<18)
#define RESULT		(1u<<19)

static const unsigned all_flags = (PARENT1 | PARENT2 | STALE | RESULT);

static int queue_has_nonstale(struct prio_queue *queue)
{
	int i;
	for (i = 0; i < queue->nr; i++) {
		struct commit *commit = queue->array[i].data;
		if (!(commit->object.flags & STALE))
			return 1;
	}
	return 0;
}

/* all input commits in one and twos[] must have been parsed! */
static struct commit_list *paint_down_to_common(struct commit *one, int n,
						struct commit **twos,
						int min_generation)
{
	struct prio_queue queue = { compare_commits_by_gen_then_commit_date };
	struct commit_list *result = NULL;
	int i;
	uint32_t last_gen = GENERATION_NUMBER_INFINITY;

	one->object.flags |= PARENT1;
	if (!n) {
		commit_list_append(one, &result);
		return result;
	}
	prio_queue_put(&queue, one);

	for (i = 0; i < n; i++) {
		twos[i]->object.flags |= PARENT2;
		prio_queue_put(&queue, twos[i]);
	}

	while (queue_has_nonstale(&queue)) {
		struct commit *commit = prio_queue_get(&queue);
		struct commit_list *parents;
		int flags;

		if (commit->generation > last_gen)
			BUG("bad generation skip %8x > %8x at %s",
			    commit->generation, last_gen,
			    oid_to_hex(&commit->object.oid));
		last_gen = commit->generation;

		if (commit->generation < min_generation)
			break;
