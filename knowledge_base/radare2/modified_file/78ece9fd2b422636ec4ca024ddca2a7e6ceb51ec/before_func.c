R_API const char* r_reg_32_to_64(RReg* reg, const char* rreg32) {
	// OMG this is shit...
	int i, j = -1;
	RListIter* iter;
	RRegItem* item;
	for (i = 0; i < R_REG_TYPE_LAST; ++i) {
		r_list_foreach (reg->regset[i].regs, iter, item) {
			if (!r_str_casecmp (rreg32, item->name) && item->size == 32) {
				j = item->offset;
				break;
			}
		}
	}
	if (j != -1) {
		for (i = 0; i < R_REG_TYPE_LAST; ++i) {
			r_list_foreach (reg->regset[i].regs, iter, item) {
				if (item->offset == j && item->size == 64) {
					return item->name;
				}
			}
		}
	}
	return NULL;
}
