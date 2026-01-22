	Error insert(Size p_pos, const T &p_val) {
		Size new_size = size() + 1;
		ERR_FAIL_INDEX_V(p_pos, new_size, ERR_INVALID_PARAMETER);
		Error err = resize(new_size);
		ERR_FAIL_COND_V(err, err);
		T *p = ptrw();
		for (Size i = new_size - 1; i > p_pos; i--) {
			p[i] = p[i - 1];
		}
		p[p_pos] = p_val;

		return OK;
	}
