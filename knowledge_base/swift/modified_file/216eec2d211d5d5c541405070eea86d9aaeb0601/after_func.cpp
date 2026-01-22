static void addMidLevelFunctionPipeline(SILPassPipelinePlan &P) {
  P.startPipeline("MidLevel,Function", true /*isFunctionPassPipeline*/);
  addFunctionPasses(P, OptimizationLevelKind::MidLevel);

  // Specialize partially applied functions with dead arguments as a preparation
  // for CapturePropagation.
  P.addDeadArgSignatureOpt();

  // A LICM pass at mid-level is mainly needed to hoist addressors of globals.
  // It needs to be before global_init functions are inlined.
  P.addLICM();
  // Run loop unrolling after inlining and constant propagation, because loop
  // trip counts may have became constant.
  P.addLICM();
  P.addLoopUnroll();
}
