static void *llist_merge(void *list, void *other,
			 void *(*get_next_fn)(const void *),
			 void (*set_next_fn)(void *, void *),
			 int (*compare_fn)(const void *, const void *))
{
	void *result = list, *tail;
	int prefer_list = compare_fn(list, other) <= 0;

	if (!prefer_list) {
		result = other;
		SWAP(list, other);
	}
	for (;;) {
		do {
			tail = list;
			list = get_next_fn(list);
			if (!list) {
				set_next_fn(tail, other);
				return result;
			}
		} while (compare_fn(list, other) < prefer_list);
		set_next_fn(tail, other);
		prefer_list ^= 1;
		SWAP(list, other);
	}
}
