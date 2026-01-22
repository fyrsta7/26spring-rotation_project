						     tmpname.buf, write_bitmap_options);
				write_bitmap_index = 0;
			}

			strbuf_release(&tmpname);
			free(pack_tmp_name);
			puts(sha1_to_hex(sha1));
		}

		/* mark written objects as written to previous pack */
		for (j = 0; j < nr_written; j++) {
			written_list[j]->offset = (off_t)-1;
		}
		nr_remaining -= nr_written;
	} while (nr_remaining && i < to_pack.nr_objects);

	free(written_list);
	free(write_order);
	stop_progress(&progress_state);
	if (written != nr_result)
		die("wrote %"PRIu32" objects while expecting %"PRIu32,
			written, nr_result);
}

static void setup_delta_attr_check(struct git_attr_check *check)
{
	static struct git_attr *attr_delta;

	if (!attr_delta)
		attr_delta = git_attr("delta");

	check[0].attr = attr_delta;
}

static int no_try_delta(const char *path)
{
	struct git_attr_check check[1];

	setup_delta_attr_check(check);
	if (git_check_attr(path, ARRAY_SIZE(check), check))
