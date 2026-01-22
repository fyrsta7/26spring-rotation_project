void EditorAssetLibrary::_update_image_queue() {

	int max_images = 2;
	int current_images = 0;

	List<int> to_delete;
	for (Map<int, ImageQueue>::Element *E = image_queue.front(); E; E = E->next()) {
		if (!E->get().active && current_images < max_images) {

			String cache_filename_base = EditorSettings::get_singleton()->get_cache_dir().plus_file("assetimage_" + E->get().image_url.md5_text());
			Vector<String> headers;

			if (FileAccess::exists(cache_filename_base + ".etag") && FileAccess::exists(cache_filename_base + ".data")) {
				FileAccess *file = FileAccess::open(cache_filename_base + ".etag", FileAccess::READ);
				if (file) {
					headers.push_back("If-None-Match: " + file->get_line());
					file->close();
					memdelete(file);
				}
			}

			Error err = E->get().request->request(E->get().image_url, headers);
			if (err != OK) {
				to_delete.push_back(E->key());
			} else {
				E->get().active = true;
			}
			current_images++;
		} else if (E->get().active) {
			current_images++;
		}
	}

	while (to_delete.size()) {
		image_queue[to_delete.front()->get()].request->queue_delete();
		image_queue.erase(to_delete.front()->get());
		to_delete.pop_front();
	}
}
