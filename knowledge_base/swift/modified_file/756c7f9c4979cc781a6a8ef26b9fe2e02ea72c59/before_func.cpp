bool SILPerformanceInliner::inlineCallsIntoFunction(SILFunction *Caller) {
  // Don't optimize functions that are marked with the opt.never attribute.
  if (!Caller->shouldOptimize())
    return false;

  // First step: collect all the functions we want to inline.  We
  // don't change anything yet so that the dominator information
  // remains valid.
  SmallVector<FullApplySite, 8> AppliesToInline;
  collectAppliesToInline(Caller, AppliesToInline);
  bool invalidatedStackNesting = false;

  if (AppliesToInline.empty())
    return false;

  // Second step: do the actual inlining.
  for (auto AI : AppliesToInline) {
    SILFunction *Callee = AI.getReferencedFunctionOrNull();
    assert(Callee && "apply_inst does not have a direct callee anymore");

    if (!Callee->shouldOptimize()) {
      continue;
    }

    // If we have a callee that doesn't have ownership, but the caller does have
    // ownership... do not inline. The two modes are incompatible, so skip this
    // apply site for now.
    if (!Callee->hasOwnership() && Caller->hasOwnership()) {
      continue;
    }

    LLVM_DEBUG(dumpCaller(Caller); llvm::dbgs()
                                   << "    inline [" << Callee->size() << "->"
                                   << Caller->size() << "] "
                                   << Callee->getName() << "\n");

    // Note that this must happen before inlining as the apply instruction
    // will be deleted after inlining.
    invalidatedStackNesting |= SILInliner::invalidatesStackNesting(AI);

    // We've already determined we should be able to inline this, so
    // unconditionally inline the function.
    //
    // If for whatever reason we can not inline this function, inlineFullApply
    // will assert, so we are safe making this assumption.
    SILInliner::inlineFullApply(AI, SILInliner::InlineKind::PerformanceInline,
                                FuncBuilder);
    NumFunctionsInlined++;
  }
  // The inliner splits blocks at call sites. Re-merge trivial branches to
  // reestablish a canonical CFG.
  mergeBasicBlocks(Caller);

  if (invalidatedStackNesting) {
    StackNesting().correctStackNesting(Caller);
  }

  // If we were asked to verify our caller after inlining all callees we could
  // find into it, do so now. This makes it easier to catch verification bugs in
  // the inliner without running the entire inliner.
  if (EnableVerifyAfterInlining) {
    Caller->verify();
  }

  return true;
}
