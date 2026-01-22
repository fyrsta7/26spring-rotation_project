void swift::runSILOptimizationPasses(SILModule &Module) {
  // Verify the module, if required.
  if (Module.getOptions().VerifyAll)
    Module.verify();

  if (Module.getOptions().DebugSerialization) {
    SILPassManager PM(&Module);
    PM.addSILLinker();
    PM.run();
    return;
  }

  SILPassManager PM(&Module, "PreSpecialize");

  // Start by cloning functions from stdlib.
  PM.addSILLinker();
  // Get rid of apparently dead functions as soon as possible so that
  // we do not spend time optimizing them.
  PM.addDeadFunctionElimination();
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

  // Do the first stack promotion on high-level SIL.
  PM.addStackPromotion();

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
  PM.addLetPropertiesOpt();

  // Propagate constants into closures and convert to static dispatch.  This
  // should run after specialization and inlining because we don't want to
  // specialize a call that can be inlined. It should run before
  // ClosureSpecialization, because constant propagation is more effective.  At
  // least one round of SSA optimization and inlining should run after this to
  // take advantage of static dispatch.
  PM.addCapturePropagation();

  // Specialize closure.
  PM.addClosureSpecializer();

  // Do the second stack promotion on low-level SIL.
  PM.addStackPromotion();

  // Speculate virtual call targets.
  PM.addSpeculativeDevirtualization();

  // We do this late since it is a pass like the inline caches that we only want
  // to run once very late. Make sure to run at least one round of the ARC
  // optimizer after this.
  PM.addFunctionSignatureOpts();

  PM.runOneIteration();
  PM.resetAndRemoveTransformations();

  // Run another iteration of the SSA optimizations to optimize the
  // devirtualized inline caches and constants propagated into closures
  // (CapturePropagation).

  PM.setStageName("LowLevel");

  // Should be after FunctionSignatureOpts and before the last inliner.
  PM.addReleaseDevirtualizer();

  // Run the devirtualizer, specializer, and inliner. If any of these
  // makes a change we'll end up restarting the function passes on the
  // current function (after optimizing any new callees).
  PM.addDevirtualizer();
  PM.addGenericSpecializer();
  PM.addLateInliner();
  AddSimplifyCFGSILCombine(PM);
  PM.addAllocBoxToStack();

  // The ReleaseDevirtualizer + specialization (in the inliner) can produce
  // aggregates in specialized deinit functions, which can be lowered.
  PM.addLowerAggregateInstrs();

  PM.addSROA();
  PM.addMem2Reg();
  PM.addCSE();
  PM.addSILCombine();
  PM.addJumpThreadSimplifyCFG();
  PM.addRedundantLoadElimination();
  PM.addDeadStoreElimination();
  PM.addCSE();
  PM.addLateCodeMotion();
  PM.addARCSequenceOpts();

  PM.runOneIteration();
  PM.resetAndRemoveTransformations();

  PM.setStageName("LateLoopOpt");

  // Delete dead code and drop the bodies of shared functions.
  PM.addExternalFunctionDefinitionsElimination();
  PM.addDeadFunctionElimination();

  // Perform the final lowering transformations.
  PM.addCodeSinking();
  PM.addLICM();

  // Optimize overflow checks.
  PM.addRedundantOverflowCheckRemoval();
  PM.addMergeCondFails();

  // Remove dead code.
  PM.addDCE();
  PM.addSimplifyCFG();
  PM.runOneIteration();

  // Call the CFG viewer.
  if (SILViewCFG) {
    PM.resetAndRemoveTransformations();
    PM.addCFGPrinter();
    PM.runOneIteration();
  }

  // Verify the module, if required.
  if (Module.getOptions().VerifyAll)
    Module.verify();
  else {
    DEBUG(Module.verify());
  }
}
