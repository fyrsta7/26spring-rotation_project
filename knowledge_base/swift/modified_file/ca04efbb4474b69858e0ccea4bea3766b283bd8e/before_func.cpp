static void addClosureSpecializePassPipeline(SILPassPipelinePlan &P) {
  P.startPipeline("ClosureSpecialize");
  P.addDeadFunctionElimination();
  P.addDeadObjectElimination();

  // Hoist globals out of loops.
  // Global-init functions should not be inlined GlobalOpt is done.
  P.addGlobalOpt();
  P.addLetPropertiesOpt();

  // Propagate constants into closures and convert to static dispatch.  This
  // should run after specialization and inlining because we don't want to
  // specialize a call that can be inlined. It should run before
  // ClosureSpecialization, because constant propagation is more effective.  At
  // least one round of SSA optimization and inlining should run after this to
  // take advantage of static dispatch.
  P.addCapturePropagation();

  // Specialize closure.
  P.addClosureSpecializer();

  // Do the second stack promotion on low-level SIL.
  P.addStackPromotion();

  // Speculate virtual call targets.
  P.addSpeculativeDevirtualization();

  // There should be at least one SILCombine+SimplifyCFG between the
  // ClosureSpecializer, etc. and the last inliner. Cleaning up after these
  // passes can expose more inlining opportunities.
  addSimplifyCFGSILCombinePasses(P);

  // We do this late since it is a pass like the inline caches that we only want
  // to run once very late. Make sure to run at least one round of the ARC
  // optimizer after this.
}
