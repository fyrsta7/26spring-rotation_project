void ARCRegionState::summarize(
    LoopRegionFunctionInfo *LRFI,
    llvm::DenseMap<const LoopRegion *, ARCRegionState *> &RegionStateInfo) {
  const LoopRegion *R = getRegion();

  // We do not need to summarize a function since it is the outermost loop.
  if (R->isFunction())
    return;

  assert(R->isLoop() && "Expected to be called on a loop");
  // Make sure that all subregions that are blocked are summarized. We know that
  // all subloops have already been summarized.
  for (unsigned SubregionID : R->getSubregions()) {
    auto *Subregion = LRFI->getRegion(SubregionID);
    if (!Subregion->isBlock())
      continue;
    auto *SubregionState = RegionStateInfo[Subregion];
    SubregionState->summarizeBlock(Subregion->getBlock());
  }

  summarizeLoop(R, LRFI, RegionStateInfo);
}
