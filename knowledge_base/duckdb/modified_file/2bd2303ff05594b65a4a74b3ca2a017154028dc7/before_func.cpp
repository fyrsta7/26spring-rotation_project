	void ScanPartial(Vector &result, idx_t result_offset, idx_t to_scan) {
		auto &result_mask = FlatVector::Validity(result);

		// This method assumes that the validity mask starts off as having all bits set for the entries that are being
		// scanned.

		if (INVERTED) {
			do {
				if (run_index >= count || runs[run_index].start > scanned_count + to_scan) {
					// The run does not cover these entries, no action required
					break;
				}
				idx_t result_idx = 0;
				while (run_index < count && result_idx < to_scan) {
					auto run = runs[run_index];

					idx_t to_skip = 0;
					if (run.start > scanned_count + result_idx) {
						to_skip = MinValue(run.start - (scanned_count + result_idx), to_scan - result_idx);
						// These entries are set, no action required
					}
					result_idx += to_skip;
					idx_t total = MinValue<idx_t>(1 + run.length, to_scan - result_idx);
					if (!to_skip) {
						total -= (scanned_count + result_idx - run.start);
					}

					// Process the run
					for (idx_t i = 0; i < total; i++) {
						result_mask.SetInvalid(result_offset + result_idx++);
					}
					if (scanned_count + result_idx == run.start + 1 + run.length) {
						// Fully processed the current run
						run_index++;
					}
				}
			} while (false);
			scanned_count += to_scan;
		} else {
			do {
				idx_t i = 0;
				while (i < to_scan) {
					// Determine the next valid position within the scan range, if available
					idx_t valid_pos = (run_index < count) ? runs[run_index].start : scanned_count + to_scan;
					idx_t valid_start = valid_pos - scanned_count;

					if (i < valid_start) {
						// FIXME: optimize this to group the SetInvalid calls
						// These bits are all set to 0
						idx_t invalid_end = MinValue<idx_t>(valid_start, to_scan);
						for (idx_t j = i; j < invalid_end; j++) {
							result_mask.SetInvalid(result_offset + j);
						}
						i = invalid_end;
					}

					if (i == valid_start && i < to_scan && run_index < count) {
						idx_t valid_end = MinValue<idx_t>(i + 1 + runs[run_index].length, to_scan);
						// These bits are already set, no action required
						i = valid_end;
						run_index++;
					}
				}
			} while (false);
			scanned_count += to_scan;
		}
	}
