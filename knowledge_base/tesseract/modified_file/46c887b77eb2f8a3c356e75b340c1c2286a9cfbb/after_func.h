  PointerVector<T>& operator=(const PointerVector& other) {
    if (&other != this) {
      this->truncate(0);
      this->operator+=(other);
    }
    return *this;
  }
