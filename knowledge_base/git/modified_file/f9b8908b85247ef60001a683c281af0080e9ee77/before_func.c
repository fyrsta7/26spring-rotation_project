					 struct commit **twos)
{
	return get_merge_bases_many_0(one, n, twos, 1);
}

struct commit_list *get_merge_bases_many_dirty(struct commit *one,
					       int n,
					       struct commit **twos)
{
	return get_merge_bases_many_0(one, n, twos, 0);
}

struct commit_list *get_merge_bases(struct commit *one, struct commit *two)
{
	return get_merge_bases_many_0(one, 1, &two, 1);
}

/*
 * Is "commit" a descendant of one of the elements on the "with_commit" list?
