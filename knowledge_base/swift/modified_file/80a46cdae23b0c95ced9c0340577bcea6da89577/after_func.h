  bool isUniquelyReferencedOrPinned() const {
    auto value = __atomic_load_n(&refCount, __ATOMIC_RELAXED);
    // Rotating right by one sets the sign bit to the pinned bit.
    // The dealloc flag is the least significant bit followed by the reference
    // count. A reference count of two or higher means that our value is bigger
    // than 3 if the pinned bit is not set. If the pinned bit is set the value
    // is negative.
    auto rotateRightByOne = ((value >> 1) | (value << (sizeof(value) * 8 - 1)));
    return ((int32_t)rotateRightByOne) < 4;
  }
