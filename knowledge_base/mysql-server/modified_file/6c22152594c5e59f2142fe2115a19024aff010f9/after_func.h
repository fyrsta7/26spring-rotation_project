      m_is_alloced = false;
    }
    return *this;
  }
  String &operator=(String &&s) noexcept {
    if (&s != this) {
