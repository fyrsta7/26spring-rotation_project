		return;
	ce->ce_flags &= ~CE_HASHED;
	hashmap_remove(&istate->name_hash, ce, ce);

	if (ignore_case)
		remove_dir_entry(istate, ce);
}

static int slow_same_name(const char *name1, int len1, const char *name2, int len2)
{
	if (len1 != len2)
		return 0;

