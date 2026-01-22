  void push_back(T&& item) {
    if (num_stack_items_ < kSize) {
      values_[num_stack_items_++] = std::move(item);
    } else {
      vect_.push_back(item);
    }
  }
