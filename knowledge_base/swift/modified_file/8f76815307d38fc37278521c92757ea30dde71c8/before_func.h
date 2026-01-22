  void replaceValueUsesWith(SILValue oldValue, SILValue newValue) {
    wereAnyCallbacksInvoked = true;

    while (!oldValue->use_empty()) {
      auto *use = *oldValue->use_begin();
      setUseValue(use, newValue);
    }
  }
