Node *Node::get_node(const NodePath &p_path) const {
	Node *node = get_node_or_null(p_path);

	if (unlikely(!node)) {
		if (p_path.is_absolute()) {
			ERR_FAIL_V_MSG(nullptr,
					vformat(R"(Node not found: "%s" (absolute path attempted from "%s").)", p_path, get_path()));
		} else {
			ERR_FAIL_V_MSG(nullptr,
					vformat(R"(Node not found: "%s" (relative to "%s").)", p_path, get_path()));
		}
	}

	return node;
}
