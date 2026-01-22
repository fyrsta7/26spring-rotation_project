void TopLevelLiveRange::AddUsePosition(UsePosition* use_pos, Zone* zone,
                                       bool trace_alloc) {
  TRACE_COND(trace_alloc, "Add to live range %d use position %d\n", vreg(),
             use_pos->pos().value());
  UsePositionVector::const_iterator insert_it = std::upper_bound(
      positions_.begin(), positions_.end(), use_pos, UsePosition::Ordering());
  // Since we `ProcessInstructions` in reverse, `positions_` are mostly
  // inserted at the front, hence grow towards that direction exclusively.
  positions_.insert<kFront>(zone, insert_it, use_pos);

  positions_span_ = base::VectorOf(positions_);
  // We must not have child `LiveRange`s yet (e.g. from splitting), otherwise we
  // would have to adjust their `positions_span_` as well.
  DCHECK_NULL(next_);
}
