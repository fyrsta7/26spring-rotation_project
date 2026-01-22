	Error insert(Size p_pos, const T &p_val) {
		ERR_FAIL_INDEX_V(p_pos, size() + 1, ERR_INVALID_PARAMETER);
		resize(size() + 1);
		for (Size i = (size() - 1); i > p_pos; i--) {
			set(i, get(i - 1));
		}
		set(p_pos, p_val);

		return OK;
	}
