void ResourceImporterScene::get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset) const {
	r_options->push_back(ImportOption(PropertyInfo(Variant::STRING, "nodes/root_type", PROPERTY_HINT_TYPE_STRING, "Node"), "Node3D"));
	r_options->push_back(ImportOption(PropertyInfo(Variant::STRING, "nodes/root_name"), "Scene Root"));

	List<String> script_extentions;
	ResourceLoader::get_recognized_extensions_for_type("Script", &script_extentions);

	String script_ext_hint;

	for (const String &E : script_extentions) {
		if (!script_ext_hint.is_empty()) {
			script_ext_hint += ",";
		}
		script_ext_hint += "*." + E;
	}

	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "nodes/root_scale", PROPERTY_HINT_RANGE, "0.001,1000,0.001"), 1.0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "meshes/ensure_tangents"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "meshes/generate_lods"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "meshes/create_shadow_meshes"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "meshes/light_baking", PROPERTY_HINT_ENUM, "Disabled,Static (VoxelGI/SDFGI),Static Lightmaps (VoxelGI/SDFGI/LightmapGI),Dynamic (VoxelGI only)", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED), 1));
	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "meshes/lightmap_texel_size", PROPERTY_HINT_RANGE, "0.001,100,0.001"), 0.2));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "skins/use_named_skins"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "animation/import"), true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::FLOAT, "animation/fps", PROPERTY_HINT_RANGE, "1,120,1"), 30));
	r_options->push_back(ImportOption(PropertyInfo(Variant::STRING, "import_script/path", PROPERTY_HINT_FILE, script_ext_hint), ""));

	r_options->push_back(ImportOption(PropertyInfo(Variant::DICTIONARY, "_subresources", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), Dictionary()));

	for (int i = 0; i < post_importer_plugins.size(); i++) {
		post_importer_plugins.write[i]->get_import_options(p_path, r_options);
	}

	for (Ref<EditorSceneFormatImporter> importer_elem : importers) {
		importer_elem->get_import_options(p_path, r_options);
	}
}
