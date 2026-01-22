		stride = 20;
	} else {
		stride = 24;
		index += 4;
	}

	if (debug_lookup)
		printf("%02x%02x%02x... lo %u hi %u nr %"PRIu32"\n",
		       sha1[0], sha1[1], sha1[2], lo, hi, p->num_objects);

	if (use_lookup < 0)
