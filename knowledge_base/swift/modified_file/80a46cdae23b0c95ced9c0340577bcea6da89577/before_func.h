  bool isUniquelyReferencedOrPinned() const {
    auto value = __atomic_load_n(&refCount, __ATOMIC_RELAXED);
    return (value >> RC_FLAGS_COUNT == 1 || value & RC_PINNED_FLAG);
  }
