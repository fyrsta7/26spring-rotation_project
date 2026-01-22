void swift::runSILOptimizationPasses(SILModule &Module) {
  if (Module.getOptions().DebugSerialization) {
    SILPassManager PM(&Module);
    registerAnalysisPasses(PM);
    PM.add(createSILLinker());
    PM.run();
    return;
  }

  SILPassManager PM(&Module, "PreSpecialize");
  registerAnalysisPasses(PM);

  // Start by specializing generics and by cloning functions from stdlib.
  PM.add(createSILLinker());
  PM.add(createGenericSpecializer());
  PM.run();
  PM.resetAndRemoveTransformations();

  // Run two iterations of the high-level SSA passes.
  PM.setStageName("HighLevel");
  AddSSAPasses(PM, OptimizationLevelKind::HighLevel);
  PM.runOneIteration();
  PM.runOneIteration();
  PM.resetAndRemoveTransformations();


  PM.setStageName("EarlyLoopOpt");
  AddHighLevelLoopOptPasses(PM);
  PM.runOneIteration();
  PM.resetAndRemoveTransformations();


  // Run two iterations of the mid-level SSA passes.
  PM.setStageName("MidLevel");
  AddSSAPasses(PM, OptimizationLevelKind::MidLevel);
  PM.runOneIteration();
  PM.runOneIteration();
  PM.resetAndRemoveTransformations();

  // Perform lowering optimizations.
  PM.setStageName("Lower");
  PM.add(createDeadFunctionElimination());
  PM.add(createDeadObjectElimination());

  // Hoist globals out of loops.
  // Global-init functions should not be inlined GlobalOpt is done.
  PM.add(createGlobalOpt());

  // Propagate constants into closures and convert to static dispatch.  This
  // should run after specialization and inlining because we don't want to
  // specialize a call that can be inlined. It should run before
  // ClosureSpecialization, because constant propagation is more effective.  At
  // least one round of SSA optimization and inlining should run after this to
  // take advantage of static dispatch.
  PM.add(createCapturePropagation());

  // Specialize closure.
  PM.add(createClosureSpecializer());

  // Insert inline caches for virtual calls.
  PM.add(createDevirtualizer());
  PM.add(createInlineCaches());

  // Optimize function signatures if we are asked to.
  //
  // We do this late since it is a pass like the inline caches that we only want
  // to run once very late. Make sure to run at least one round of the ARC
  // optimizer after this.
  if (Module.getOptions().EnableFuncSigOpts)
    PM.add(createFunctionSignatureOpts());

  PM.runOneIteration();
  PM.resetAndRemoveTransformations();

  // Run another iteration of the SSA optimizations to optimize the
  // devirtualized inline caches and constants propagated into closures
  // (CapturePropagation).

  PM.setStageName("LowLevel");
  AddSSAPasses(PM, OptimizationLevelKind::LowLevel);
  PM.runOneIteration();
  PM.resetAndRemoveTransformations();

  PM.setStageName("LateLoopOpt");
  AddLowLevelLoopOptPasses(PM);

  // Perform the final lowering transformations.
  PM.add(createExternalFunctionDefinitionsElimination());
  PM.add(createDeadFunctionElimination());
  PM.add(createMergeCondFails());
  PM.add(createCropOverflowChecks());
  PM.runOneIteration();

  // Call the CFG viewer.
  if (SILViewCFG) {
    PM.resetAndRemoveTransformations();
    PM.add(createCFGPrinter());
    PM.runOneIteration();
  }

  DEBUG(Module.verify());
}
