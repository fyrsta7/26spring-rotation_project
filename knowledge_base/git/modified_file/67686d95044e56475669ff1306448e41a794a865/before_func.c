		break;
	case OBJ_TREE:
		strcpy(type, "tree");
		break;
	case OBJ_BLOB:
		strcpy(type, "blob");
		break;
	case OBJ_TAG:
		strcpy(type, "tag");
		break;
	default:
		die("corrupted pack file %s containing object of kind %d",
		    p->pack_name, kind);
	}
	*store_size = 0; /* notyet */
}

static int packed_object_info(struct pack_entry *entry,
			      char *type, unsigned long *sizep)
{
	struct packed_git *p = entry->p;
	unsigned long offset, size, left;
	unsigned char *pack;
	enum object_type kind;
	int retval;

	if (use_packed_git(p))
		die("cannot map packed file");

	offset = unpack_object_header(p, entry->offset, &kind, &size);
	pack = p->pack_base + offset;
	left = p->pack_size - offset;

	switch (kind) {
	case OBJ_DELTA:
		retval = packed_delta_info(pack, size, left, type, sizep, p);
		unuse_packed_git(p);
		return retval;
	case OBJ_COMMIT:
		strcpy(type, "commit");
		break;
	case OBJ_TREE:
		strcpy(type, "tree");
		break;
	case OBJ_BLOB:
		strcpy(type, "blob");
		break;
	case OBJ_TAG:
		strcpy(type, "tag");
		break;
