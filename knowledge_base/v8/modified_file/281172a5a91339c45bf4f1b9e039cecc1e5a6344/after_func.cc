void OperandAssigner::AssignSpillSlots() {
  ZoneVector<SpillRange*> spill_ranges(data()->allocation_zone());
  for (const TopLevelLiveRange* range : data()->live_ranges()) {
    if (range != nullptr && range->HasSpillRange()) {
      DCHECK_NOT_NULL(range->GetSpillRange());
      spill_ranges.push_back(range->GetSpillRange());
    }
  }
  // At this point, the `SpillRange`s for all `TopLevelLiveRange`s should be
  // unique, since none have been merged yet.
  DCHECK_EQ(spill_ranges.size(),
            std::set(spill_ranges.begin(), spill_ranges.end()).size());

  // Merge all `SpillRange`s that belong to the same `LiveRangeBundle`.
  for (const TopLevelLiveRange* range : data()->live_ranges()) {
    data()->tick_counter()->TickAndMaybeEnterSafepoint();
    if (range != nullptr && range->get_bundle() != nullptr) {
      range->get_bundle()->MergeSpillRangesAndClear();
    }
  }

  // Now merge *all* disjoint, non-empty `SpillRange`s.
  // Formerly, this merging was O(n^2) in the number of `SpillRange`s, which
  // then dominated compile time (>40%) for some pathological cases,
  // e.g., https://crbug.com/v8/14133.
  // Now, we allow only `kMaxRetries` unsuccessful merges with directly
  // following `SpillRange`s. After each `kMaxRetries`, we exponentially
  // increase the stride, which limits the inner loop to O(log n) and thus
  // the overall merging to O(n * log n).

  // The merging above may have left some `SpillRange`s empty, remove them.
  SpillRange** end_nonempty =
      std::remove_if(spill_ranges.begin(), spill_ranges.end(),
                     [](const SpillRange* range) { return range->IsEmpty(); });
  for (SpillRange** range_it = spill_ranges.begin(); range_it < end_nonempty;
       ++range_it) {
    data()->tick_counter()->TickAndMaybeEnterSafepoint();
    SpillRange* range = *range_it;
    DCHECK(!range->IsEmpty());
    constexpr size_t kMaxRetries = 1000;
    size_t retries = kMaxRetries;
    size_t stride = 1;
    for (SpillRange** other_it = range_it + 1; other_it < end_nonempty;
         other_it += stride) {
      SpillRange* other = *other_it;
      DCHECK(!other->IsEmpty());
      if (range->TryMerge(other)) {
        DCHECK(other->IsEmpty());
        std::iter_swap(other_it, --end_nonempty);
      } else if (--retries == 0) {
        retries = kMaxRetries;
        stride *= 2;
      }
    }
  }
  spill_ranges.erase(end_nonempty, spill_ranges.end());

  // Allocate slots for the merged spill ranges.
  for (SpillRange* range : spill_ranges) {
    data()->tick_counter()->TickAndMaybeEnterSafepoint();
    DCHECK(!range->IsEmpty());
    if (!range->HasSlot()) {
      // Allocate a new operand referring to the spill slot, aligned to the
      // operand size.
      int width = range->byte_width();
      int index = data()->frame()->AllocateSpillSlot(width, width);
      range->set_assigned_slot(index);
    }
  }
}
