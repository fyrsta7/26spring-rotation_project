R_API void r_anal_trace_bb(RAnal *anal, ut64 addr) {
	r_return_if_fail (anal);
	RAnalBlock *bb = r_anal_get_block_at (anal, addr);
	if (bb && !bb->traced) {
		bb->traced = true;
		R_DIRTY (anal);
	}
}
