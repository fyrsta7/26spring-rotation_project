{
	struct object_cb_data *data = vdata;
	oidcpy(&data->expand->oid, oid);
	batch_object_write(NULL, data->opt, data->expand);
	return 0;
}

static int collect_loose_object(const struct object_id *oid,
				const char *path,
