R_API bool r_anal_block_op_starts_at(RAnalBlock *bb, ut64 addr) {
	if (!r_anal_block_contains (bb, addr)) {
		return false;
	}
	ut64 off = addr - bb->addr;
	if (off == 0) {
		return true;
	}
	if (off > UT16_MAX) {
		return false;
	}
	if (bb->ninstr < 1) {
		return true;
	}
	size_t i;
	for (i = 0; i < bb->ninstr; i++) {
		ut16 inst_off = r_anal_bb_offset_inst (bb, i);
		if (inst_off > off) {
			break;
		}
		if (off == inst_off) {
			return true;
		}
	}
	return false;
}
