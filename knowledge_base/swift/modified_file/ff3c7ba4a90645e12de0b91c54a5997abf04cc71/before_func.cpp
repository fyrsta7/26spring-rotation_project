void addFunctionPasses(SILPassPipelinePlan &P,
                       OptimizationLevelKind OpLevel) {
  // Promote box allocations to stack allocations.
  P.addAllocBoxToStack();

  // Propagate copies through stack locations.  Should run after
  // box-to-stack promotion since it is limited to propagating through
  // stack locations. Should run before aggregate lowering since that
  // splits up copy_addr.
  P.addCopyForwarding();

  // Optimize copies from a temporary (an "l-value") to a destination.
  P.addTempLValueOpt();

  // Split up opaque operations (copy_addr, retain_value, etc.).
  P.addLowerAggregateInstrs();

  // Split up operations on stack-allocated aggregates (struct, tuple).
  if (OpLevel == OptimizationLevelKind::HighLevel) {
    P.addEarlySROA();
  } else {
    P.addSROA();
  }

  // Promote stack allocations to values.
  P.addMem2Reg();

  // Run the existential specializer Pass.
  P.addExistentialSpecializer();

  // Cleanup, which is important if the inliner has restarted the pass pipeline.
  P.addPerformanceConstantPropagation();

  addSimplifyCFGSILCombinePasses(P);

  P.addArrayElementPropagation();

  // Perform a round of loop/array optimization in the mid-level pipeline after
  // potentially inlining semantic calls, e.g. Array append. The high level
  // pipeline only optimizes semantic calls *after* inlining (see
  // addHighLevelLoopOptPasses). For example, the high-level pipeline may
  // perform ArrayElementPropagation and after inlining a level of semantic
  // calls, the mid-level pipeline may handle uniqueness hoisting. Do this as
  // late as possible before inlining because it must run between runs of the
  // inliner when the pipeline restarts.
  if (OpLevel == OptimizationLevelKind::MidLevel) {
    P.addHighLevelLICM();
    P.addArrayCountPropagation();
    P.addABCOpt();
    P.addDCE();
    P.addCOWArrayOpts();
    P.addDCE();
    P.addSwiftArrayPropertyOpt();
  }

  // Run the devirtualizer, specializer, and inliner. If any of these
  // makes a change we'll end up restarting the function passes on the
  // current function (after optimizing any new callees).
  P.addDevirtualizer();
  P.addGenericSpecializer();
  // Run devirtualizer after the specializer, because many
  // class_method/witness_method instructions may use concrete types now.
  P.addDevirtualizer();

  // We earlier eliminated ownership if we are not compiling the stdlib. Now
  // handle the stdlib functions, re-simplifying, eliminating ARC as we do.
  P.addCopyPropagation();
  P.addSemanticARCOpts();
  P.addNonTransparentFunctionOwnershipModelEliminator();

  switch (OpLevel) {
  case OptimizationLevelKind::HighLevel:
    // Does not inline functions with defined semantics.
    P.addEarlyInliner();
    break;
  case OptimizationLevelKind::MidLevel:
    // Does inline semantics-functions (except "availability"), but not
    // global-init functions.
    P.addPerfInliner();
    break;
  case OptimizationLevelKind::LowLevel:
    // Inlines everything
    P.addLateInliner();
    break;
  }

  // Promote stack allocations to values and eliminate redundant
  // loads.
  P.addMem2Reg();
  P.addPerformanceConstantPropagation();
  //  Do a round of CFG simplification, followed by peepholes, then
  //  more CFG simplification.

  // Jump threading can expose opportunity for SILCombine (enum -> is_enum_tag->
  // cond_br).
  P.addJumpThreadSimplifyCFG();
  P.addPhiExpansion();
  P.addSILCombine();
  // SILCombine can expose further opportunities for SimplifyCFG.
  P.addSimplifyCFG();

  P.addCSE();
  if (OpLevel == OptimizationLevelKind::HighLevel) {
    // Early RLE does not touch loads from Arrays. This is important because
    // later array optimizations, like ABCOpt, get confused if an array load in
    // a loop is converted to a pattern with a phi argument.
    P.addEarlyRedundantLoadElimination();
  } else {
    P.addRedundantLoadElimination();
  }
  // Optimize copies created during RLE.
  P.addSemanticARCOpts();

  P.addCOWOpts();
  P.addPerformanceConstantPropagation();
  // Remove redundant arguments right before CSE and DCE, so that CSE and DCE
  // can cleanup redundant and dead instructions.
  P.addRedundantPhiElimination();
  P.addCSE();
  P.addDCE();

  // Perform retain/release code motion and run the first ARC optimizer.
  P.addEarlyCodeMotion();
  P.addReleaseHoisting();
  P.addARCSequenceOpts();
  P.addTempRValueOpt();

  P.addSimplifyCFG();
  if (OpLevel == OptimizationLevelKind::LowLevel) {
    // Remove retain/releases based on Builtin.unsafeGuaranteed
    P.addUnsafeGuaranteedPeephole();
    // Only hoist releases very late.
    P.addLateCodeMotion();
  } else
    P.addEarlyCodeMotion();

  P.addRetainSinking();
  // Retain sinking does not sink all retains in one round.
  // Let it run one more time time, because it can be beneficial.
  // FIXME: Improve the RetainSinking pass to sink more/all
  // retains in one go.
  P.addRetainSinking();
  P.addReleaseHoisting();
  P.addARCSequenceOpts();
}
