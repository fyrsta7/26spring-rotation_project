static int process(struct object *obj)
{
	if (obj->flags & SEEN)
		return 0;
	obj->flags |= SEEN;

	if (has_sha1_file(obj->sha1)) {
		/* We already have it, so we should scan it now. */
		obj->flags |= TO_SCAN;
	} else {
		if (obj->flags & COMPLETE)
			return 0;
		prefetch(obj->sha1);
	}
		
	object_list_insert(obj, process_queue_end);
	process_queue_end = &(*process_queue_end)->next;
	return 0;
}
