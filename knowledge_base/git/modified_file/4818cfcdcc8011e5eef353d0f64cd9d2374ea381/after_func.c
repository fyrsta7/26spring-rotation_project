struct object *lookup_object(const unsigned char *sha1)
{
	unsigned int i, first;
	struct object *obj;

	if (!obj_hash)
		return NULL;

	first = i = hashtable_index(sha1);
	while ((obj = obj_hash[i]) != NULL) {
		if (!hashcmp(sha1, obj->sha1))
			break;
		i++;
		if (i == obj_hash_size)
			i = 0;
	}
	if (obj && i != first) {
		/*
		 * Move object to where we started to look for it so
		 * that we do not need to walk the hash table the next
		 * time we look for it.
		 */
		struct object *tmp = obj_hash[i];
		obj_hash[i] = obj_hash[first];
