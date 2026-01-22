bool SILPerformanceInliner::inlineCallsIntoFunction(SILFunction *Caller) {
  bool Changed = false;
  SILInliner Inliner(*Caller, SILInliner::InlineKind::PerformanceInline);

  DEBUG(llvm::dbgs() << "Visiting Function: " << Caller->getName() << "\n");

  llvm::SmallVector<ApplyInst*, 8> CallSites;

  // Collect all of the ApplyInsts in this function. We will be changing the
  // control flow and collecting the AIs simplifies the scan.
  for (auto &BB : *Caller) {
    auto I = BB.begin(), E = BB.end();
    while (I != E) {
      // Check if this is a call site.
      ApplyInst *AI = dyn_cast<ApplyInst>(I++);
      if (AI)
        CallSites.push_back(AI);
    }
  }

  for (auto AI : CallSites) {
    DEBUG(llvm::dbgs() << "  Found call site:" <<  *AI);

    // Get the callee.
    SILFunction *Callee = getInlinableFunction(AI, Mode);
    if (!Callee)
      continue;

    DEBUG(llvm::dbgs() << "  Found callee:" <<  Callee->getName() << ".\n");

    // Prevent circular inlining.
    if (Callee == Caller) {
      DEBUG(llvm::dbgs() << "  Skipping recursive calls.\n");
      continue;
    }

    // If Callee has a less visible linkage than caller or references something
    // with a less visible linkage than caller, don't inline Callee into caller.
    if (transitivelyReferencesLessVisibleLinkage(*Callee,
                                                 Caller->getLinkage())) {
      DEBUG(llvm::dbgs() << "  Skipping less visible call.");
      continue;
    }

    // Check if the function takes a closure.
    bool HasClosure = false;
    for (auto &Op : AI->getAllOperands()) {
      if (isa<PartialApplyInst>(Op.get().getDef())) {
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

    if (CalleeCost > InlineCostThreshold * BoostFactor) {
      DEBUG(llvm::dbgs() << "  Function too big to inline. Skipping.\n");
      continue;
    }

    // Add the arguments from AI into a SILValue list.
    SmallVector<SILValue, 8> Args;
    for (const auto &Arg : AI->getArguments())
    Args.push_back(Arg);

    // Ok, we are within budget. Attempt to inline.
    DEBUG(llvm::dbgs() << "  Inlining " << Callee->getName() << " Into " <<
          Caller->getName() << "\n");

    // We already moved the iterator to the next instruction because the AI
    // will be erased by the inliner. Notice that we will skip all of the
    // newly inlined ApplyInsts. That's okay because we will visit them in
    // our next invocation of the inliner.
    Inliner.inlineFunction(AI, Callee, ArrayRef<Substitution>(), Args);
    NumFunctionsInlined++;
    Changed = true;
  }
  return Changed;
}
