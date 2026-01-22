struct object *lookup_object(const unsigned char *sha1)
{
	unsigned int i;
	struct object *obj;

	if (!obj_hash)
		return NULL;

	i = hashtable_index(sha1);
	while ((obj = obj_hash[i]) != NULL) {
		if (!hashcmp(sha1, obj->sha1))
			break;
		i++;
		if (i == obj_hash_size)
			i = 0;
	}
	return obj;
}
