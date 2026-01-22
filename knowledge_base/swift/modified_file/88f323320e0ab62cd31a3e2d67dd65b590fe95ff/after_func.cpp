static unsigned functionInlineCost(SILFunction *F) {
  if (F->isTransparent() == IsTransparent_t::IsTransparent)
    return 0;

  unsigned i = 0;
  for (auto &BB : *F) {
    for (auto &I : BB) {
      i += instructionInlineCost(I);

      // If i is greater than the InlineCostThreshold, we already know we are
      // not going to inline this given function, so there is no point in
      // continuing to visit instructions.
      if (i > InlineCostThreshold)
        return i;
    }
  }
  return i;
}
