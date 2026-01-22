  template <typename T> void AssignToString(T* str) {
    uint64_t ofs = start_ofs();
    str->clear();
    str->reserve(Length());
    while (ofs < end_ofs()) {
      size_t len;
      const char *ptr = GetPtr(ofs, &len);
      str->append(ptr, len);
      ofs += len;
    }
  }
