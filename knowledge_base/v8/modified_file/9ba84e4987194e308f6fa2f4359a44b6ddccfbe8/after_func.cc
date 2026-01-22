void TopLevelLiveRange::AddUsePosition(UsePosition* use_pos, Zone* zone,
                                       bool trace_alloc) {
  TRACE_COND(trace_alloc, "Add to live range %d use position %d\n", vreg(),
             use_pos->pos().value());
  // Since we `ProcessInstructions` in reverse, the `use_pos` is almost always
  // inserted at the front of `positions_`, hence (i) use linear instead of
  // binary search and (ii) grow towards the `kFront` exclusively on `insert`.
  UsePositionVector::iterator insert_it = std::find_if(
      positions_.begin(), positions_.end(), [=](const UsePosition* pos) {
        return UsePosition::Ordering()(use_pos, pos);
      });
  positions_.insert<kFront>(zone, insert_it, use_pos);

  positions_span_ = base::VectorOf(positions_);
  // We must not have child `LiveRange`s yet (e.g. from splitting), otherwise we
  // would have to adjust their `positions_span_` as well.
  DCHECK_NULL(next_);
}
