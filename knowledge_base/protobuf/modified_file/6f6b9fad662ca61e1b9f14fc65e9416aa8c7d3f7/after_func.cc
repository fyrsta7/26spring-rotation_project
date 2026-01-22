template <typename F>
void* RepeatedPtrFieldBase::AddInternal(F factory) {
  Arena* const arena = GetArena();
  if (tagged_rep_or_elem_ == nullptr) {
    ExchangeCurrentSize(1);
    tagged_rep_or_elem_ = factory(arena);
    return tagged_rep_or_elem_;
  }
  absl::PrefetchToLocalCache(tagged_rep_or_elem_);
  if (using_sso()) {
    if (current_size_ == 0) {
      ExchangeCurrentSize(1);
      return tagged_rep_or_elem_;
    }
    void*& result = *InternalExtend(1);
    result = factory(arena);
    Rep* r = rep();
    r->allocated_size = 2;
    ExchangeCurrentSize(2);
    return result;
  }
  Rep* r = rep();
  if (PROTOBUF_PREDICT_FALSE(SizeAtCapacity())) {
    InternalExtend(1);
    r = rep();
  } else {
    if (current_size_ != r->allocated_size) {
      return r->elements[ExchangeCurrentSize(current_size_ + 1)];
    }
  }
  ++r->allocated_size;
  void*& result = r->elements[ExchangeCurrentSize(current_size_ + 1)];
  result = factory(arena);
  return result;
}
