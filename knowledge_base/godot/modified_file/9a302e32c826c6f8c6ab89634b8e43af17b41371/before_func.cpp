void ScriptEditor::_close_tab(int p_idx, bool p_save, bool p_history_back) {
	int selected = p_idx;
	if (selected < 0 || selected >= tab_container->get_tab_count()) {
		return;
	}

	Node *tselected = tab_container->get_tab_control(selected);

	ScriptEditorBase *current = Object::cast_to<ScriptEditorBase>(tselected);
	if (current) {
		Ref<Resource> file = current->get_edited_resource();
		if (p_save && file.is_valid()) {
			// Do not try to save internal scripts, but prompt to save in-memory
			// scripts which are not saved to disk yet (have empty path).
			if (!file->is_built_in()) {
				save_current_script();
			}
		}
		if (file.is_valid()) {
			if (!file->get_path().is_empty()) {
				// Only saved scripts can be restored.
				previous_scripts.push_back(file->get_path());
			}

			Ref<Script> scr = file;
			if (scr.is_valid()) {
				notify_script_close(scr);
			}
		}
	}

	// roll back to previous tab
	if (p_history_back) {
		_history_back();
	}

	//remove from history
	history.resize(history_pos + 1);

	for (int i = 0; i < history.size(); i++) {
		if (history[i].control == tselected) {
			history.remove_at(i);
			i--;
			history_pos--;
		}
	}

	if (history_pos >= history.size()) {
		history_pos = history.size() - 1;
	}

	int idx = tab_container->get_current_tab();
	if (current) {
		current->clear_edit_menu();
		_save_editor_state(current);
	}
	memdelete(tselected);
	if (idx >= tab_container->get_tab_count()) {
		idx = tab_container->get_tab_count() - 1;
	}
	if (idx >= 0) {
		if (history_pos >= 0) {
			idx = tab_container->get_tab_idx_from_control(history[history_pos].control);
		}
		_go_to_tab(idx);
	} else {
		_update_selected_editor_menu();
	}

	_update_history_arrows();

	_update_script_names();
	_update_members_overview_visibility();
	_update_help_overview_visibility();
	_save_layout();
	_update_find_replace_bar();
}
