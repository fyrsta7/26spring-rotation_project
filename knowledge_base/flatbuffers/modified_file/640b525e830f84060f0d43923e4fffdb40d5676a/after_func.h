  }

  uint8_t *ReserveElements(size_t len, size_t elemsize) {
    return buf_.make_space(len * elemsize);
  }
  /// @endcond

  /// @brief Serialize an array into a FlatBuffer `vector`.
  /// @tparam T The data type of the array elements.
  /// @param[in] v A pointer to the array of type `T` to serialize into the
  /// buffer as a `vector`.
