void Node::replace_by(Node *p_node, bool p_keep_groups) {
	ERR_THREAD_GUARD
	ERR_FAIL_NULL(p_node);
	ERR_FAIL_COND(p_node->data.parent);

	List<Node *> owned = data.owned;
	List<Node *> owned_by_owner;
	Node *owner = (data.owner == this) ? p_node : data.owner;

	if (p_keep_groups) {
		List<GroupInfo> groups;
		get_groups(&groups);

		for (const GroupInfo &E : groups) {
			p_node->add_to_group(E.name, E.persistent);
		}
	}

	_replace_connections_target(p_node);

	if (data.owner) {
		for (int i = 0; i < get_child_count(); i++) {
			find_owned_by(data.owner, get_child(i), &owned_by_owner);
		}

		_clean_up_owner();
	}

	Node *parent = data.parent;
	int index_in_parent = get_index(false);

	if (data.parent) {
		parent->remove_child(this);
		parent->add_child(p_node);
		parent->move_child(p_node, index_in_parent);
	}

	emit_signal(SNAME("replacing_by"), p_node);

	while (get_child_count()) {
		Node *child = get_child(0);
		remove_child(child);
		if (!child->is_owned_by_parent()) {
			// add the custom children to the p_node
			p_node->add_child(child);
		}
	}

	p_node->set_owner(owner);
	for (int i = 0; i < owned.size(); i++) {
		owned[i]->set_owner(p_node);
	}

	for (int i = 0; i < owned_by_owner.size(); i++) {
		owned_by_owner[i]->set_owner(owner);
	}

	p_node->set_scene_file_path(get_scene_file_path());
}