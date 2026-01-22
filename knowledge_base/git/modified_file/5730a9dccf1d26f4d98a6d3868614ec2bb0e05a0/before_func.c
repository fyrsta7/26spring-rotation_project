{
	if (iterator_is_null(&mi->stack[idx]))
		return 0;
	return merged_iter_advance_nonnull_subiter(mi, idx);
}

static int merged_iter_next_entry(struct merged_iter *mi,
				  struct reftable_record *rec)
{
	struct pq_entry entry = { 0 };
	int err = 0;

	if (merged_iter_pqueue_is_empty(mi->pq))
		return 1;

	entry = merged_iter_pqueue_remove(&mi->pq);
	err = merged_iter_advance_subiter(mi, entry.index);
	if (err < 0)
		return err;

	/*
	  One can also use reftable as datacenter-local storage, where the ref
	  database is maintained in globally consistent database (eg.
	  CockroachDB or Spanner). In this scenario, replication delays together
	  with compaction may cause newer tables to contain older entries. In
	  such a deployment, the loop below must be changed to collect all
	  entries for the same key, and return new the newest one.
	*/
	while (!merged_iter_pqueue_is_empty(mi->pq)) {
		struct pq_entry top = merged_iter_pqueue_top(mi->pq);
		int cmp;

		cmp = reftable_record_cmp(&top.rec, &entry.rec);
		if (cmp > 0)
			break;

		merged_iter_pqueue_remove(&mi->pq);
		err = merged_iter_advance_subiter(mi, top.index);
