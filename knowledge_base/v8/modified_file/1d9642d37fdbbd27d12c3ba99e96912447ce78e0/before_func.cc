void TopLevelLiveRange::Splinter(LifetimePosition start, LifetimePosition end,
                                 TopLevelLiveRange* result, Zone* zone) {
  DCHECK(start != Start() || end != End());
  DCHECK(start < end);

  result->set_spill_type(spill_type());

  if (start <= Start()) {
    DCHECK(end < End());
    DetachAt(end, result, zone);
    next_ = nullptr;
  } else if (end >= End()) {
    DCHECK(start > Start());
    DetachAt(start, result, zone);
    next_ = nullptr;
  } else {
    DCHECK(start < End() && Start() < end);

    const int kInvalidId = std::numeric_limits<int>::max();

    DetachAt(start, result, zone);

    LiveRange end_part(kInvalidId, this->machine_type(), nullptr);
    result->DetachAt(end, &end_part, zone);

    next_ = end_part.next_;
    last_interval_->set_next(end_part.first_interval_);
    last_interval_ = end_part.last_interval_;


    if (first_pos_ == nullptr) {
      first_pos_ = end_part.first_pos_;
    } else {
      UsePosition* pos = first_pos_;
      for (; pos->next() != nullptr; pos = pos->next()) {
      }
      pos->set_next(end_part.first_pos_);
    }
  }
  result->next_ = nullptr;
  result->top_level_ = result;

  result->SetSplinteredFrom(this);
}
