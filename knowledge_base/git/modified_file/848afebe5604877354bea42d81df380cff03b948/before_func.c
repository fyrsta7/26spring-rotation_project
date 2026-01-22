static void *llist_merge(void *list, void *other,
			 void *(*get_next_fn)(const void *),
			 void (*set_next_fn)(void *, void *),
			 int (*compare_fn)(const void *, const void *))
{
	void *result = list, *tail;

	if (compare_fn(list, other) > 0) {
		result = other;
		goto other;
	}
	for (;;) {
		do {
			tail = list;
			list = get_next_fn(list);
			if (!list) {
				set_next_fn(tail, other);
				return result;
			}
		} while (compare_fn(list, other) <= 0);
		set_next_fn(tail, other);
	other:
		do {
			tail = other;
			other = get_next_fn(other);
			if (!other) {
				set_next_fn(tail, list);
				return result;
			}
		} while (compare_fn(list, other) > 0);
		set_next_fn(tail, list);
	}
}
