void GreedyAllocator::SplitAndSpillRangesDefinedByMemoryOperand() {
  size_t initial_range_count = data()->live_ranges().size();
  for (size_t i = 0; i < initial_range_count; ++i) {
    auto range = data()->live_ranges()[i];
    if (!CanProcessRange(range)) continue;
    if (range->HasNoSpillType()) continue;

    LifetimePosition start = range->Start();
    TRACE("Live range %d:%d is defined by a spill operand.\n",
          range->TopLevel()->vreg(), range->relative_id());
    auto next_pos = start;
    if (next_pos.IsGapPosition()) {
      next_pos = next_pos.NextStart();
    }
    auto pos = range->NextUsePositionRegisterIsBeneficial(next_pos);
    // If the range already has a spill operand and it doesn't need a
    // register immediately, split it and spill the first part of the range.
    if (pos == nullptr) {
      Spill(range);
    } else if (pos->pos() > range->Start().NextStart()) {
      // Do not spill live range eagerly if use position that can benefit from
      // the register is too close to the start of live range.
      auto split_pos = pos->pos();
      if (data()->IsBlockBoundary(split_pos.Start())) {
        split_pos = split_pos.Start();
      } else {
        split_pos = split_pos.PrevStart().End();
      }
      Split(range, data(), split_pos);
      Spill(range);
    }
  }
}
