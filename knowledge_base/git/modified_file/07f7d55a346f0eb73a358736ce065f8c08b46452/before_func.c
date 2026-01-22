			commit_list_insert(parents->item, plist);

		commit = commit->parents->item;
	}
}

void clear_commit_marks_many(int nr, struct commit **commit, unsigned int mark)
{
	struct commit_list *list = NULL;

	while (nr--) {
