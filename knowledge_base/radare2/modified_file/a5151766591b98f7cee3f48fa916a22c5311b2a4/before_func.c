R_API int r_isprint(const RRune c) {
	const int last = nonprintable_ranges_count;

	int low = 0;
	int hi = last - 1;

	do {
		int mid = (low + hi) >> 1;
		if (c >= nonprintable_ranges[mid].from && c <= nonprintable_ranges[mid].to) {
			return false;
		}
		if (mid < last && c > nonprintable_ranges[mid].to) {
			low = mid + 1;
		}
		if (mid < last && c < nonprintable_ranges[mid].from) {
			hi = mid - 1;
		}
	} while (low <= hi);

	return true;
}
