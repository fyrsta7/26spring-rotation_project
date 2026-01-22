static int flag_skiplist_cmp(const void *va, const void *vb) {
	const RFlagsAtOffset *a = (RFlagsAtOffset *)va, *b = (RFlagsAtOffset *)vb;
	if (a->off == b->off) {
		return 0;
	}
	return a->off < b->off? -1: 1;
}
