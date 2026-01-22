Ref<Texture2D> EditorNode::get_class_icon(const String &p_class, const String &p_fallback) const {
	ERR_FAIL_COND_V_MSG(p_class.empty(), nullptr, "Class name cannot be empty.");

	if (gui_base->has_theme_icon(p_class, "EditorIcons")) {
		return gui_base->get_theme_icon(p_class, "EditorIcons");
	}

	if (ScriptServer::is_global_class(p_class)) {
		Ref<ImageTexture> icon;
		Ref<Script> script = EditorNode::get_editor_data().script_class_load_script(p_class);

		while (script.is_valid()) {
			StringName name = EditorNode::get_editor_data().script_class_get_name(script->get_path());
			String current_icon_path = EditorNode::get_editor_data().script_class_get_icon_path(name);
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
