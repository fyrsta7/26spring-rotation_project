static bool
runOnFunctionRecursively(SILFunction *F, ApplyInst* AI,
                         DenseFunctionSet &FullyInlinedSet,
                         ImmutableFunctionSet::Factory &SetFactory,
                         ImmutableFunctionSet CurrentInliningSet) {
  // Prevent attempt to circularly inline.
  if (CurrentInliningSet.contains(F)) {
    // This cannot happen on a top-level call, so AI should be non-null.
    assert(AI && "Cannot have circular inline without apply");
    SILLocation L = AI->getLoc();
    assert(L && "Must have location for forced inline apply");
    diagnose(F->getModule().getASTContext(), L.getStartSourceLoc(),
             diag::circular_force_inline);
    return false;
  }

  // Add to the current inlining set (immutably, so we only affect the set
  // during this call and recursive subcalls).
  CurrentInliningSet = SetFactory.add(CurrentInliningSet, F);

  SmallVector<ApplyInst*, 4> ApplySites;
  for (auto &BB : *F) {
    for (auto &I : BB) {
      ApplyInst *InnerAI;
      if ((InnerAI = dyn_cast<ApplyInst>(&I)) && InnerAI->isForceInline()) {
        // Figure out of this is something we have the body for
        // FIXME: once fragile SIL is serialized in modules, these can be
        // asserts, since forced inline functions should always have their
        // bodies available.
        SILValue Callee = InnerAI->getCallee();
        FunctionRefInst *FRI = dyn_cast<FunctionRefInst>(Callee.getDef());
        if (!FRI)
          continue;
        assert(Callee.getResultNumber() == 0);
        SILFunction *CalledFunc = FRI->getFunction();
        if (!CalledFunc || CalledFunc->empty())
          continue;

        // If we haven't fully processed this function yet, then recursively
        // process it first before trying to inline it.
        if (FullyInlinedSet.find(CalledFunc) == FullyInlinedSet.end() &&
            !runOnFunctionRecursively(CalledFunc, InnerAI, FullyInlinedSet,
                                      SetFactory, CurrentInliningSet)) {
          // If we failed due to circular inlining, then emit some notes to
          // trace back the failure if we have more information.
          // FIXME: possibly it could be worth recovering and attempting other
          // inlines within this same recursive call rather than simply
          // propogating the failure.
          if (AI) {
            SILLocation L = AI->getLoc();
            assert(L && "Must have location for forced inline apply");
            diagnose(F->getModule().getASTContext(), L.getStartSourceLoc(),
                     diag::note_while_inlining);
          }
          return false;
        }

        ApplySites.push_back(InnerAI);
      }
    }
  }

  // Do the inlining separately from the inspection loop to avoid iterator
  // invalidation issues.
  if (!ApplySites.empty()) {
    SILInliner Inliner(*F);
    for (auto *InnerAI : ApplySites) {
      Inliner.inlineFunction(InnerAI);
      ++NumMandatoryInlineSitesInlined;
    }
  }

  // Keep track of full inlined functions so we don't waste time recursively
  // reprocessing them.
  FullyInlinedSet.insert(F);
  return true;
}
