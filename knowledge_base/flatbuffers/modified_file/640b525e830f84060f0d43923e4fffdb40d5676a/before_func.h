  }

  uint8_t *ReserveElements(size_t len, size_t elemsize) {
    return buf_.make_space(len * elemsize);
  }
  /// @endcond

