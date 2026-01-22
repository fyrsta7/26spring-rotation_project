static bool addMidLevelPassPipeline(SILPassPipelinePlan &P) {
  P.startPipeline("MidLevel");
  addSSAPasses(P, OptimizationLevelKind::MidLevel);
  if (P.getOptions().StopOptimizationAfterSerialization)
    return true;

  // Specialize partially applied functions with dead arguments as a preparation
  // for CapturePropagation.
  P.addDeadArgSignatureOpt();

  // Run loop unrolling after inlining and constant propagation, because loop
  // trip counts may have became constant.
  P.addLoopUnroll();
  return false;
}
