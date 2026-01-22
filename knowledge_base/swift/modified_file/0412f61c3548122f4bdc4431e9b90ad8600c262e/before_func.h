  template<typename T>
  llvm::Error writeRaw(uint32_t Offset, T Value) {
    if (auto Error = checkOffsetForWrite(Offset, sizeof(T))) {
      return Error;
    }

    // Resize the internal buffer if needed.
    uint32_t RequiredSize = Offset + sizeof(T);
    if (RequiredSize > Data.size()) {
      Data.resize(RequiredSize);
    }

    *(T *)(Data.data() + Offset) = Value;

    return llvm::Error::success();
  }
