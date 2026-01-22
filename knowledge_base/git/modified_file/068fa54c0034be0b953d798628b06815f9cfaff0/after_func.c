			     struct commit **commits,
			     uint32_t commits_nr,
			     uint32_t *pack_order,
			     const char *refs_snapshot,
			     unsigned flags)
{
	int ret, i;
	uint16_t options = 0;
	struct pack_idx_entry **index;
	char *bitmap_name = xstrfmt("%s-%s.bitmap", midx_name,
					hash_to_hex(midx_hash));

	if (flags & MIDX_WRITE_BITMAP_HASH_CACHE)
		options |= BITMAP_OPT_HASH_CACHE;

	/*
	 * Build the MIDX-order index based on pdata.objects (which is already
	 * in MIDX order; c.f., 'midx_pack_order_cmp()' for the definition of
	 * this order).
	 */
	ALLOC_ARRAY(index, pdata->nr_objects);
	for (i = 0; i < pdata->nr_objects; i++)
		index[i] = &pdata->objects[i].idx;

	bitmap_writer_show_progress(flags & MIDX_PROGRESS);
	bitmap_writer_build_type_index(pdata, index, pdata->nr_objects);

	/*
	 * bitmap_writer_finish expects objects in lex order, but pack_order
	 * gives us exactly that. use it directly instead of re-sorting the
	 * array.
	 *
	 * This changes the order of objects in 'index' between
	 * bitmap_writer_build_type_index and bitmap_writer_finish.
	 *
	 * The same re-ordering takes place in the single-pack bitmap code via
	 * write_idx_file(), which is called by finish_tmp_packfile(), which
	 * happens between bitmap_writer_build_type_index() and
	 * bitmap_writer_finish().
	 */
	for (i = 0; i < pdata->nr_objects; i++)
		index[pack_order[i]] = &pdata->objects[i].idx;

	bitmap_writer_select_commits(commits, commits_nr, -1);
	ret = bitmap_writer_build(pdata);
	if (ret < 0)
		goto cleanup;

	bitmap_writer_set_checksum(midx_hash);
	bitmap_writer_finish(index, pdata->nr_objects, bitmap_name, options);

cleanup:
	free(index);
	free(bitmap_name);
	return ret;
}

static struct multi_pack_index *lookup_multi_pack_index(struct repository *r,
							const char *object_dir)
{
	struct multi_pack_index *result = NULL;
	struct multi_pack_index *cur;
	char *obj_dir_real = real_pathdup(object_dir, 1);
	struct strbuf cur_path_real = STRBUF_INIT;

	/* Ensure the given object_dir is local, or a known alternate. */
	find_odb(r, obj_dir_real);

	for (cur = get_multi_pack_index(r); cur; cur = cur->next) {
		strbuf_realpath(&cur_path_real, cur->object_dir, 1);
		if (!strcmp(obj_dir_real, cur_path_real.buf)) {
			result = cur;
			goto cleanup;
		}
	}

cleanup:
	free(obj_dir_real);
	strbuf_release(&cur_path_real);
	return result;
}

static int write_midx_internal(const char *object_dir,
			       struct string_list *packs_to_include,
			       struct string_list *packs_to_drop,
			       const char *preferred_pack_name,
			       const char *refs_snapshot,
			       unsigned flags)
{
	struct strbuf midx_name = STRBUF_INIT;
	unsigned char midx_hash[GIT_MAX_RAWSZ];
	uint32_t i;
	struct hashfile *f = NULL;
	struct lock_file lk;
	struct write_midx_context ctx = { 0 };
	int pack_name_concat_len = 0;
	int dropped_packs = 0;
	int result = 0;
	struct chunkfile *cf;

	get_midx_filename(&midx_name, object_dir);
	if (safe_create_leading_directories(midx_name.buf))
		die_errno(_("unable to create leading directories of %s"),
			  midx_name.buf);

	if (!packs_to_include) {
		/*
		 * Only reference an existing MIDX when not filtering which
		 * packs to include, since all packs and objects are copied
		 * blindly from an existing MIDX if one is present.
		 */
		ctx.m = lookup_multi_pack_index(the_repository, object_dir);
	}

	if (ctx.m && !midx_checksum_valid(ctx.m)) {
		warning(_("ignoring existing multi-pack-index; checksum mismatch"));
		ctx.m = NULL;
	}

	ctx.nr = 0;
	ctx.alloc = ctx.m ? ctx.m->num_packs : 16;
	ctx.info = NULL;
	ALLOC_ARRAY(ctx.info, ctx.alloc);

	if (ctx.m) {
		for (i = 0; i < ctx.m->num_packs; i++) {
			ALLOC_GROW(ctx.info, ctx.nr + 1, ctx.alloc);

			ctx.info[ctx.nr].orig_pack_int_id = i;
			ctx.info[ctx.nr].pack_name = xstrdup(ctx.m->pack_names[i]);
			ctx.info[ctx.nr].p = ctx.m->packs[i];
			ctx.info[ctx.nr].expired = 0;

			if (flags & MIDX_WRITE_REV_INDEX) {
				/*
				 * If generating a reverse index, need to have
				 * packed_git's loaded to compare their
				 * mtimes and object count.
				 */
				if (prepare_midx_pack(the_repository, ctx.m, i)) {
					error(_("could not load pack"));
					result = 1;
					goto cleanup;
				}

				if (open_pack_index(ctx.m->packs[i]))
					die(_("could not open index for %s"),
					    ctx.m->packs[i]->pack_name);
				ctx.info[ctx.nr].p = ctx.m->packs[i];
			}

			ctx.nr++;
		}
	}

	ctx.pack_paths_checked = 0;
	if (flags & MIDX_PROGRESS)
		ctx.progress = start_delayed_progress(_("Adding packfiles to multi-pack-index"), 0);
	else
		ctx.progress = NULL;

	ctx.to_include = packs_to_include;

	for_each_file_in_pack_dir(object_dir, add_pack_to_midx, &ctx);
	stop_progress(&ctx.progress);

	if ((ctx.m && ctx.nr == ctx.m->num_packs) &&
	    !(packs_to_include || packs_to_drop)) {
		struct bitmap_index *bitmap_git;
		int bitmap_exists;
		int want_bitmap = flags & MIDX_WRITE_BITMAP;

		bitmap_git = prepare_midx_bitmap_git(ctx.m);
		bitmap_exists = bitmap_git && bitmap_is_midx(bitmap_git);
		free_bitmap_index(bitmap_git);

		if (bitmap_exists || !want_bitmap) {
			/*
			 * The correct MIDX already exists, and so does a
			 * corresponding bitmap (or one wasn't requested).
			 */
			if (!want_bitmap)
				clear_midx_files_ext(object_dir, ".bitmap",
						     NULL);
			goto cleanup;
		}
	}

	if (preferred_pack_name) {
		int found = 0;
		for (i = 0; i < ctx.nr; i++) {
			if (!cmp_idx_or_pack_name(preferred_pack_name,
						  ctx.info[i].pack_name)) {
				ctx.preferred_pack_idx = i;
				found = 1;
				break;
			}
		}

		if (!found)
			warning(_("unknown preferred pack: '%s'"),
				preferred_pack_name);
	} else if (ctx.nr &&
		   (flags & (MIDX_WRITE_REV_INDEX | MIDX_WRITE_BITMAP))) {
		struct packed_git *oldest = ctx.info[ctx.preferred_pack_idx].p;
		ctx.preferred_pack_idx = 0;

		if (packs_to_drop && packs_to_drop->nr)
			BUG("cannot write a MIDX bitmap during expiration");

		/*
		 * set a preferred pack when writing a bitmap to ensure that
		 * the pack from which the first object is selected in pseudo
		 * pack-order has all of its objects selected from that pack
		 * (and not another pack containing a duplicate)
		 */
		for (i = 1; i < ctx.nr; i++) {
			struct packed_git *p = ctx.info[i].p;

			if (!oldest->num_objects || p->mtime < oldest->mtime) {
				oldest = p;
				ctx.preferred_pack_idx = i;
			}
		}

		if (!oldest->num_objects) {
			/*
			 * If all packs are empty; unset the preferred index.
			 * This is acceptable since there will be no duplicate
			 * objects to resolve, so the preferred value doesn't
			 * matter.
			 */
			ctx.preferred_pack_idx = -1;
		}
	} else {
		/*
		 * otherwise don't mark any pack as preferred to avoid
		 * interfering with expiration logic below
		 */
		ctx.preferred_pack_idx = -1;
	}

	if (ctx.preferred_pack_idx > -1) {
		struct packed_git *preferred = ctx.info[ctx.preferred_pack_idx].p;
		if (!preferred->num_objects) {
			error(_("cannot select preferred pack %s with no objects"),
			      preferred->pack_name);
			result = 1;
			goto cleanup;
		}
	}

	ctx.entries = get_sorted_entries(ctx.m, ctx.info, ctx.nr, &ctx.entries_nr,
					 ctx.preferred_pack_idx);

	ctx.large_offsets_needed = 0;
	for (i = 0; i < ctx.entries_nr; i++) {
		if (ctx.entries[i].offset > 0x7fffffff)
			ctx.num_large_offsets++;
		if (ctx.entries[i].offset > 0xffffffff)
			ctx.large_offsets_needed = 1;
	}

	QSORT(ctx.info, ctx.nr, pack_info_compare);

	if (packs_to_drop && packs_to_drop->nr) {
		int drop_index = 0;
		int missing_drops = 0;

		for (i = 0; i < ctx.nr && drop_index < packs_to_drop->nr; i++) {
			int cmp = strcmp(ctx.info[i].pack_name,
					 packs_to_drop->items[drop_index].string);

			if (!cmp) {
				drop_index++;
				ctx.info[i].expired = 1;
			} else if (cmp > 0) {
				error(_("did not see pack-file %s to drop"),
				      packs_to_drop->items[drop_index].string);
				drop_index++;
				missing_drops++;
				i--;
			} else {
				ctx.info[i].expired = 0;
			}
		}

		if (missing_drops) {
			result = 1;
			goto cleanup;
		}
	}

	/*
	 * pack_perm stores a permutation between pack-int-ids from the
	 * previous multi-pack-index to the new one we are writing:
	 *
	 * pack_perm[old_id] = new_id
	 */
	ALLOC_ARRAY(ctx.pack_perm, ctx.nr);
	for (i = 0; i < ctx.nr; i++) {
		if (ctx.info[i].expired) {
			dropped_packs++;
			ctx.pack_perm[ctx.info[i].orig_pack_int_id] = PACK_EXPIRED;
		} else {
			ctx.pack_perm[ctx.info[i].orig_pack_int_id] = i - dropped_packs;
		}
	}

	for (i = 0; i < ctx.nr; i++) {
		if (!ctx.info[i].expired)
			pack_name_concat_len += strlen(ctx.info[i].pack_name) + 1;
	}

	/* Check that the preferred pack wasn't expired (if given). */
	if (preferred_pack_name) {
		struct pack_info *preferred = bsearch(preferred_pack_name,
						      ctx.info, ctx.nr,
						      sizeof(*ctx.info),
						      idx_or_pack_name_cmp);
		if (preferred) {
			uint32_t perm = ctx.pack_perm[preferred->orig_pack_int_id];
			if (perm == PACK_EXPIRED)
				warning(_("preferred pack '%s' is expired"),
