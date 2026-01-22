static int flag_skiplist_cmp(const void *va, const void *vb) {
	const ut64 ao = ((RFlagsAtOffset *)va)->off;
	const ut64 bo = ((RFlagsAtOffset *)vb)->off;
	if (R_LIKELY (ao < bo)) {
		return -1;
	}
	if (R_LIKELY (ao > bo)) {
		return 1;
	}
	return 0;
}
