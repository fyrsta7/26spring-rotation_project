  static WaitQueue *getAsWaitQueue(uintptr_t value) {
    if (value & IsWaitQueue)
      return reinterpret_cast<WaitQueue*>(value & ~IsWaitQueue);
    return nullptr;
  }
