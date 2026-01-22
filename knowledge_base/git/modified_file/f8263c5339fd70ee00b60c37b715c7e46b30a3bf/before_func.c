	if (!parent_name ||
	    commit_name->generation + 1 < parent_name->generation)
		name_commit(parent, commit_name->head_name,
			    commit_name->generation + 1);
}

static int name_first_parent_chain(struct commit *c)
{
	int i = 0;
	while (c) {
		struct commit *p;
		if (!c->util)
			break;
		if (!c->parents)
			break;
		p = c->parents->item;
		if (!p->util) {
			name_parent(c, p);
