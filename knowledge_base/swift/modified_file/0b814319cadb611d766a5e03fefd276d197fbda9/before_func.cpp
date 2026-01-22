void swift::runSILOptimizationPasses(SILModule &Module) {
  if (Module.getOptions().DebugSerialization) {
    SILPassManager PM(&Module);
    registerAnalysisPasses(PM);
    PM.addSILLinker();
    PM.run();
    return;
  }

  SILPassManager PM(&Module, "PreSpecialize");
  registerAnalysisPasses(PM);

  // Start by specializing generics and by cloning functions from stdlib.
  PM.addSILLinker();
  PM.addGenericSpecializer();
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
  
  PM.addDeadFunctionElimination();
  PM.addDeadObjectElimination();
  PM.addGlobalPropertyOpt();
  
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
  PM.addDeadFunctionElimination();
  PM.addDeadObjectElimination();

  // Hoist globals out of loops.
  // Global-init functions should not be inlined GlobalOpt is done.
  PM.addGlobalOpt();

  // Propagate constants into closures and convert to static dispatch.  This
  // should run after specialization and inlining because we don't want to
  // specialize a call that can be inlined. It should run before
  // ClosureSpecialization, because constant propagation is more effective.  At
  // least one round of SSA optimization and inlining should run after this to
  // take advantage of static dispatch.
  PM.addCapturePropagation();

  // Specialize closure.
  PM.addClosureSpecializer();

  // Insert inline caches for virtual calls.
  PM.addInlineCaches();

  // Optimize function signatures if we are asked to.
  //
  // We do this late since it is a pass like the inline caches that we only want
  // to run once very late. Make sure to run at least one round of the ARC
  // optimizer after this.
  if (Module.getOptions().EnableFuncSigOpts)
    PM.addFunctionSignatureOpts();

  PM.runOneIteration();
  PM.resetAndRemoveTransformations();

  // Run another iteration of the SSA optimizations to optimize the
  // devirtualized inline caches and constants propagated into closures
  // (CapturePropagation).

  PM.setStageName("LowLevel");

  PM.addLateInliner();
  AddSimplifyCFGSILCombine(PM);
  PM.addAllocBoxToStack();
  PM.addSROA();
  PM.addMem2Reg();
  PM.addCSE();
  PM.addSILCombine();
  PM.addJumpThreadSimplifyCFG();
  PM.addGlobalLoadStoreOpts();
  PM.addLateCodeMotion();
  PM.addGlobalARCOpts();

  PM.runOneIteration();
  PM.resetAndRemoveTransformations();

  PM.setStageName("LateLoopOpt");
  PM.addLICM();

  // Perform the final lowering transformations.
  PM.addExternalFunctionDefinitionsElimination();
  PM.addDeadFunctionElimination();
  PM.addMergeCondFails();
  PM.addCropOverflowChecks();
  PM.runOneIteration();

  // Call the CFG viewer.
  if (SILViewCFG) {
    PM.resetAndRemoveTransformations();
    PM.addCFGPrinter();
    PM.runOneIteration();
  }

  DEBUG(Module.verify());
}
