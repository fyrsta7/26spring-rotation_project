R_API int r_isprint(const RRune c) {
	// RRunes are most commonly single byte... We can early out with this common case.
	if (c < 0x34F) {
		/*
		manually copied from top, please update if this ever changes
		{ 0x0000, 0x001F }, { 0x007F, 0x009F }, { 0x034F, 0x034F },
		could do a linear search, but thats a lot slower than a few compare
		*/
		return !( c <= 0x1F || ( c >= 0x7F && c <= 0x9F));
	}

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
