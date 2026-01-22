  V8_NOINLINE void Grow(int slots_needed, Zone* zone) {
    size_t new_capacity = std::max(
        size_t{8}, base::bits::RoundUpToPowerOfTwo(size() + slots_needed));
    CHECK_GE(kMaxUInt32, new_capacity);
    DCHECK_LT(capacity_end_ - begin_, new_capacity);
    T* new_begin = zone->template NewArray<T>(new_capacity);
    if (begin_) {
      for (T *ptr = begin_, *new_ptr = new_begin; ptr != end_;
           ++ptr, ++new_ptr) {
        new (new_ptr) T{std::move(*ptr)};
        ptr->~T();
      }
      zone->DeleteArray(begin_, capacity_end_ - begin_);
    }
    end_ = new_begin + (end_ - begin_);
    begin_ = new_begin;
    capacity_end_ = new_begin + new_capacity;
  }
