	_FORCE_INLINE_ bool is_readable_from_caller_thread() const {
		if (current_process_thread_group == nullptr) {
			// No thread processing.
			// Only accessible if node is outside the scene tree
			// or access will happen from a node-safe thread.
			return is_current_thread_safe_for_nodes() || unlikely(!data.inside_tree);
		} else {
			// Thread processing.
			return true;
		}
	}
