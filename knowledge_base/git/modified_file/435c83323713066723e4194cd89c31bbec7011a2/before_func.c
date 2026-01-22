		for (i = 0; i < shallows.nr; i++) {
			struct object *object = shallows.objects[i].item;
			if (object->flags & NOT_SHALLOW) {
				struct commit_list *parents;
				packet_write(1, "unshallow %s",
					sha1_to_hex(object->sha1));
				object->flags &= ~CLIENT_SHALLOW;
				/* make sure the real parents are parsed */
				unregister_shallow(object->sha1);
				object->parsed = 0;
				if (parse_commit((struct commit *)object))
					die("invalid commit");
				parents = ((struct commit *)object)->parents;
				while (parents) {
					add_object_array(&parents->item->object,
							NULL, &want_obj);
					parents = parents->next;
				}
				add_object_array(object, NULL, &extra_edge_obj);
			}
			/* make sure commit traversal conforms to client */
			register_shallow(object->sha1);
		}
		packet_flush(1);
	} else
		if (shallows.nr > 0) {
			int i;
			for (i = 0; i < shallows.nr; i++)
				register_shallow(shallows.objects[i].item->sha1);
		}

	shallow_nr += shallows.nr;
	free(shallows.objects);
}
