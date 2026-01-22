	return idx_next != UT16_MAX? idx_next - idx_cur: bb->size - idx_cur;
}

/* returns the address of the basic block that contains addr or UT64_MAX if
 * there is no such basic block */
R_API ut64 r_anal_get_bbaddr(RAnal *anal, ut64 addr) {
	RAnalBlock *bb;
	RListIter *iter;
	RAnalFunction *fcni = r_anal_get_fcn_in_bounds (anal, addr, 0);
	if (fcni) {
		r_list_foreach (fcni->bbs, iter, bb) {
			if (addr >= bb->addr && addr < bb->addr + bb->size) {
				return bb->addr;
