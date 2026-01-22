  // Return true if it successfully inserts `runner`. `index` is supposed to be
  // dense.
  bool Insert(int64_t index, OpKernelRunner runner) {
    if (runners_.size() <= index) runners_.resize(index + 1);
    if (runners_[index].has_value()) return false;
    runners_[index] = std::move(runner);
    return true;
  }

  // Return the OpKernelRunner at the corresponding `index` in the table. The
