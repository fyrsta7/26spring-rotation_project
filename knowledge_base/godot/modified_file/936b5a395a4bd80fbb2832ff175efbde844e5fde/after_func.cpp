void EditorNode::_save_scene_with_preview(String p_file, int p_idx) {

	EditorProgress save("save", TTR("Saving Scene"), 4);
	save.step(TTR("Analyzing"), 0);

	int c2d = 0;
	int c3d = 0;
	_find_node_types(editor_data.get_edited_scene_root(), c2d, c3d);

	RID viewport;
	bool is2d;
	if (c3d < c2d) {
		viewport = scene_root->get_viewport_rid();
		is2d = true;
	} else {
		viewport = SpatialEditor::get_singleton()->get_editor_viewport(0)->get_viewport_node()->get_viewport_rid();
		is2d = false;
	}
	save.step(TTR("Creating Thumbnail"), 1);
	//current view?

	Ref<Image> img;
	if (is2d) {
		img = scene_root->get_texture()->get_data();
	} else {
		img = SpatialEditor::get_singleton()->get_editor_viewport(0)->get_viewport_node()->get_texture()->get_data();
	}

	if (img.is_valid()) {
		save.step(TTR("Creating Thumbnail"), 2);
		save.step(TTR("Creating Thumbnail"), 3);

		int preview_size = EditorSettings::get_singleton()->get("filesystem/file_dialog/thumbnail_size");
		preview_size *= EDSCALE;

		// consider a square region
		int vp_size = MIN(img->get_width(), img->get_height());
		int x = (img->get_width() - vp_size) / 2;
		int y = (img->get_height() - vp_size) / 2;

		if (vp_size < preview_size) {
			// just square it.
			img->crop_from_point(x, y, vp_size, vp_size);
		} else {
			int ratio = vp_size / preview_size;
			int size = preview_size * (ratio / 2);

			x = (img->get_width() - size) / 2;
			y = (img->get_height() - size) / 2;

			img->crop_from_point(x, y, size, size);
			// We could get better pictures with better filters
			img->resize(preview_size, preview_size, Image::INTERPOLATE_CUBIC);
		}
		img->convert(Image::FORMAT_RGB8);

		img->flip_y();

		//save thumbnail directly, as thumbnailer may not update due to actual scene not changing md5
		String temp_path = EditorSettings::get_singleton()->get_cache_dir();
		String cache_base = ProjectSettings::get_singleton()->globalize_path(p_file).md5_text();
		cache_base = temp_path.plus_file("resthumb-" + cache_base);

		//does not have it, try to load a cached thumbnail

		String file = cache_base + ".png";

		post_process_preview(img);
		img->save_png(file);
	}

	save.step(TTR("Saving Scene"), 4);
	_save_scene(p_file, p_idx);
	EditorResourcePreview::get_singleton()->check_for_invalidation(p_file);
}
