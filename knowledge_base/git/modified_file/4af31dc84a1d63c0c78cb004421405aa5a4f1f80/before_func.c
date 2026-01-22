	log->value.update.email =
		xstrndup(split.mail_begin, split.mail_end - split.mail_begin);
	log->value.update.time = atol(split.date_begin);
	if (*split.tz_begin == '-') {
		sign = -1;
		split.tz_begin++;
	}
	if (*split.tz_begin == '+') {
		sign = 1;
		split.tz_begin++;
	}

	log->value.update.tz_offset = sign * atoi(split.tz_begin);
}

static int read_ref_without_reload(struct reftable_stack *stack,
				   const char *refname,
				   struct object_id *oid,
				   struct strbuf *referent,
				   unsigned int *type)
{
	struct reftable_ref_record ref = {0};
	int ret;

	ret = reftable_stack_read_ref(stack, refname, &ref);
	if (ret)
		goto done;

	if (ref.value_type == REFTABLE_REF_SYMREF) {
		strbuf_reset(referent);
		strbuf_addstr(referent, ref.value.symref);
		*type |= REF_ISSYMREF;
	} else if (reftable_ref_record_val1(&ref)) {
		oidread(oid, reftable_ref_record_val1(&ref));
	} else {
		/* We got a tombstone, which should not happen. */
		BUG("unhandled reference value type %d", ref.value_type);
	}

done:
	assert(ret != REFTABLE_API_ERROR);
	reftable_ref_record_release(&ref);
	return ret;
}

static struct ref_store *reftable_be_init(struct repository *repo,
					  const char *gitdir,
					  unsigned int store_flags)
{
