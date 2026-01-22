	 */
	if (stored->bitmap)
		ewah_free(stored->bitmap);

	stored->bitmap = bitmap_to_ewah(ent->bitmap);

	hash_pos = kh_put_oid_map(writer.bitmaps, commit->object.oid, &hash_ret);
	if (hash_ret == 0)
		die("Duplicate entry when writing index: %s",
		    oid_to_hex(&commit->object.oid));
	kh_value(writer.bitmaps, hash_pos) = stored;
}

void bitmap_writer_build(struct packing_data *to_pack)
{
	struct bitmap_builder bb;
	size_t i;
	int nr_stored = 0; /* for progress */

	writer.bitmaps = kh_init_oid_map();
	writer.to_pack = to_pack;

	if (writer.show_progress)
		writer.progress = start_progress("Building bitmaps", writer.selected_nr);
	trace2_region_enter("pack-bitmap-write", "building_bitmaps_total",
			    the_repository);

	bitmap_builder_init(&bb, &writer);
	for (i = bb.commits_nr; i > 0; i--) {
		struct commit *commit = bb.commits[i-1];
		struct bb_commit *ent = bb_data_at(&bb.data, commit);
		struct commit *child;

		fill_bitmap_commit(ent, commit);

		if (ent->selected) {
			store_selected(ent, commit);
			nr_stored++;
			display_progress(writer.progress, nr_stored);
		}

		while ((child = pop_commit(&ent->children))) {
			struct bb_commit *child_ent =
				bb_data_at(&bb.data, child);

			if (child_ent->bitmap)
				bitmap_or(child_ent->bitmap, ent->bitmap);
			else
				child_ent->bitmap = bitmap_dup(ent->bitmap);
