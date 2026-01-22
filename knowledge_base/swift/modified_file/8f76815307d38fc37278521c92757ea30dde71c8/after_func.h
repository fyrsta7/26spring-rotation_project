  void replaceValueUsesWith(SILValue oldValue, SILValue newValue) {
    wereAnyCallbacksInvoked = true;

    // If setUseValueFunc is not set, just call RAUW directly. RAUW in this case
    // is equivalent to what we do below. We just enable better
    // performance. This ensures that the default InstModCallback is really
    // fast.
    if (!setUseValueFunc)
      return oldValue->replaceAllUsesWith(newValue);

    while (!oldValue->use_empty()) {
      auto *use = *oldValue->use_begin();
      setUseValue(use, newValue);
    }
  }
