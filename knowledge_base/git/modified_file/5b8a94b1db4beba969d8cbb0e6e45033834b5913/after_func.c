		break;
	case 0:
		buf = read_object_with_reference(sha1, exp_type, &size, NULL);
		break;

	default:
		die("git-cat-file: unknown option: %s\n", exp_type);
	}

	if (!buf)
		die("git-cat-file %s: bad file", obj_name);

	write_or_die(1, buf, size);
	return 0;
}

static int batch_one_object(const char *obj_name, int print_contents)
{
	unsigned char sha1[20];
	enum object_type type = 0;
	unsigned long size;
	void *contents = contents;

	if (!obj_name)
	   return 1;

	if (get_sha1(obj_name, sha1)) {
		printf("%s missing\n", obj_name);
		fflush(stdout);
		return 0;
	}

	if (print_contents == BATCH)
		contents = read_sha1_file(sha1, &type, &size);
	else
		type = sha1_object_info(sha1, &size);

	if (type <= 0) {
		printf("%s missing\n", obj_name);
