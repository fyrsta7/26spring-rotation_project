Ref<Texture2D> EditorNode::get_class_icon(const String &p_class, const String &p_fallback) const {
	ERR_FAIL_COND_V_MSG(p_class.empty(), nullptr, "Class name cannot be empty.");

	if (gui_base->has_theme_icon(p_class, "EditorIcons")) {
		return gui_base->get_theme_icon(p_class, "EditorIcons");
	}

	if (ScriptServer::is_global_class(p_class)) {
		String icon_path = EditorNode::get_editor_data().script_class_get_icon_path(p_class);
		Ref<ImageTexture> icon = _load_custom_class_icon(icon_path);
		if (icon.is_valid()) {
			return icon;
		}

		Ref<Script> script = ResourceLoader::load(ScriptServer::get_global_class_path(p_class), "Script");

		while (script.is_valid()) {
			String current_icon_path;
			script->get_language()->get_global_class_name(script->get_path(), nullptr, &current_icon_path);
			icon = _load_custom_class_icon(current_icon_path);
			if (icon.is_valid()) {
				return icon;
			}
			script = script->get_base_script();
		}

		if (icon.is_null()) {
			icon = gui_base->get_theme_icon(ScriptServer::get_global_class_base(p_class), "EditorIcons");
		}

		return icon;
	}

	const Map<String, Vector<EditorData::CustomType>> &p_map = EditorNode::get_editor_data().get_custom_types();
	for (const Map<String, Vector<EditorData::CustomType>>::Element *E = p_map.front(); E; E = E->next()) {
		const Vector<EditorData::CustomType> &ct = E->value();
		for (int i = 0; i < ct.size(); ++i) {
			if (ct[i].name == p_class) {
				if (ct[i].icon.is_valid()) {
					return ct[i].icon;
				}
			}
		}
	}

	if (p_fallback.length() && gui_base->has_theme_icon(p_fallback, "EditorIcons")) {
		return gui_base->get_theme_icon(p_fallback, "EditorIcons");
	}

	return nullptr;
}
