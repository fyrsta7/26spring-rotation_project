  template <typename T> void AssignToString(T* str) {
    uint64_t ofs = start_ofs();
    size_t len;
    const char *ptr = GetPtr(ofs, &len);
    str->assign(ptr, len);
    ofs += len;
    while (ofs < end_ofs()) {
      ptr = GetPtr(ofs, &len);
      str->append(ptr, len);
      ofs += len;
    }
  }
