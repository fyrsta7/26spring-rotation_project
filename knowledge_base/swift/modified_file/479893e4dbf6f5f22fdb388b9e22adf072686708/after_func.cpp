void ARCRegionState::summarize(
    LoopRegionFunctionInfo *LRFI,
    llvm::DenseMap<const LoopRegion *, ARCRegionState *> &RegionStateInfo) {
  const LoopRegion *R = getRegion();

  // We do not need to summarize a function since it is the outermost loop.
  if (R->isFunction())
    return;

  assert(R->isLoop() && "Expected to be called on a loop");
  // We know that all of our sub blocks have the correct interesting insts since
  // we did one scan at the beginning and are updating our interesting inst list
  // as we move around retains/releases. Additionally since we are going through
  // the loop nest bottom up, all of our subloops have already been
  // summarized. Thus all we need to do is gather up the interesting
  // instructions from our subregions.
  summarizeLoop(R, LRFI, RegionStateInfo);
}
