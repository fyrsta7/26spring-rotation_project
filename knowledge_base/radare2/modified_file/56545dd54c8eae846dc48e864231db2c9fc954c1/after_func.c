	return idx_next != UT16_MAX? idx_next - idx_cur: bb->size - idx_cur;
}

/* returns the address of the basic block that contains addr or UT64_MAX if
