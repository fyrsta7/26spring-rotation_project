bool AliasAnalysis::canApplyDecrementRefCount(FullApplySite FAS, SILValue Ptr) {
  // Treat applications of no-return functions as decrementing ref counts. This
  // causes the apply to become a sink barrier for ref count increments.
  if (FAS.isCalleeNoReturn())
    return true;

  /// If the pointer cannot escape to the function we are done.
  if (!EA->canEscapeTo(Ptr, FAS))
    return false;

  FunctionSideEffects ApplyEffects;
  SEA->getCalleeEffects(ApplyEffects, FAS);

  auto &GlobalEffects = ApplyEffects.getGlobalEffects();
  if (ApplyEffects.mayReadRC() || GlobalEffects.mayRelease())
    return true;

  /// The function has no unidentified releases, so let's look at the arguments
  // in detail.
  for (unsigned Idx = 0, End = FAS.getNumArguments(); Idx < End; ++Idx) {
    auto &ArgEffect = ApplyEffects.getParameterEffects()[Idx];
    if (ArgEffect.mayRelease()) {
      // The function may release this argument, so check if the pointer can
      // escape to it.
      if (EA->mayReleaseContent(FAS.getArgument(Idx), Ptr))
        return true;
    }
  }
  return false;
}
