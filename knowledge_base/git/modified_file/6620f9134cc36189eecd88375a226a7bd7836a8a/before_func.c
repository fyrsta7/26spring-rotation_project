	const struct reftable_ref_record *rec =
		(const struct reftable_ref_record *)r;
	strbuf_reset(dest);
	strbuf_addstr(dest, rec->refname);
}

static void reftable_ref_record_copy_from(void *rec, const void *src_rec,
					  int hash_size)
{
	struct reftable_ref_record *ref = rec;
	const struct reftable_ref_record *src = src_rec;
	assert(hash_size > 0);

	/* This is simple and correct, but we could probably reuse the hash
	 * fields. */
	reftable_ref_record_release(ref);
	if (src->refname) {
		ref->refname = xstrdup(src->refname);
	}
	ref->update_index = src->update_index;
	ref->value_type = src->value_type;
	switch (src->value_type) {
	case REFTABLE_REF_DELETION:
		break;
	case REFTABLE_REF_VAL1:
		memcpy(ref->value.val1, src->value.val1, hash_size);
		break;
	case REFTABLE_REF_VAL2:
		memcpy(ref->value.val2.value, src->value.val2.value, hash_size);
		memcpy(ref->value.val2.target_value,
