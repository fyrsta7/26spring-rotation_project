R_API void r_anal_trace_bb(RAnal *anal, ut64 addr) {
	r_return_if_fail (anal);
	RAnalBlock *bbi;
	RListIter *iter2;
	RAnalFunction *fcni = r_anal_get_fcn_in (anal, addr, 0);
	if (fcni) {
		r_list_foreach (fcni->bbs, iter2, bbi) {
			if (addr >= bbi->addr && addr < (bbi->addr + bbi->size)) {
				bbi->traced = true;
				break;
			}
		}
	}
	R_DIRTY (anal);
}
