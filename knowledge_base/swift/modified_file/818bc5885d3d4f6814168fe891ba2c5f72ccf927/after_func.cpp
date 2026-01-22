bool SILPerformanceInliner::isProfitableToInline(SILFunction *Caller,
                                                 SILFunction *Callee,
                                                 const ApplyInst *AI,
                                                 unsigned CalleeCount) {
  /// Always inline transparent calls. This should have been done during
  /// MandatoryInlining, but generics are not currenly handled.
  if (AI->isTransparent())
    return true;

  // To handle recursion and prevent massive code size expansion, we prevent
  // inlining the same callee many times into the caller. The recursion
  // detection logic in CallGraphAnalysis can't handle class_method in the
  // callee. To avoid inlining the recursion too many times, we stop at the
  // threshold (currently set to 1024).
  const unsigned CallsToCalleeThreshold = 1024;
  if (CalleeCount > CallsToCalleeThreshold) {
    DEBUG(llvm::dbgs() <<
          "        FAIL! Skipping callees that are called too many times.\n");
    return false;
  }

  // Prevent circular inlining.
  if (Callee == Caller) {
    DEBUG(llvm::dbgs() << "        FAIL! Skipping recursive calls.\n");
    return false;
  }

  // If Callee has a less visible linkage than caller or references something
  // with a less visible linkage than caller, don't inline Callee into caller.
  if (transitivelyReferencesLessVisibleLinkage(*Callee,
                                               Caller->getLinkage())) {
    DEBUG(llvm::dbgs() << "        FAIL! Skipping less visible call.");
    return false;
  }

  // Check if the function takes a closure.
  bool HasClosure = false;
  for (auto &Op : AI->getAllOperands()) {
    if (isa<PartialApplyInst>(Op.get())) {
      HasClosure = true;
      break;
    }
  }
  // If the function accepts a closure increase the threshold because
  // inlining has the potential to eliminate the closure.
  unsigned BoostFactor = HasClosure ? 2 : 1;

  // Calculate the inlining cost of the callee.
  unsigned CalleeCost = getFunctionCost(Callee, Caller,
                                        InlineCostThreshold * BoostFactor);

  unsigned Threshold = InlineCostThreshold * BoostFactor;
  if (CalleeCost > Threshold) {
    DEBUG(llvm::dbgs() << "        FAIL! Function too big to inline. "
          "Skipping. CalleeCost: " << CalleeCost << ". Threshold: "
          << Threshold << "\n");
    return false;
  }
  return true;
}
