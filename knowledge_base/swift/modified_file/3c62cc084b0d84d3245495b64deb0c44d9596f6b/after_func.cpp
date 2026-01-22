void addSSAPasses(SILPassPipelinePlan &P, OptimizationLevelKind OpLevel) {
  // Promote box allocations to stack allocations.
  P.addAllocBoxToStack();

  // Propagate copies through stack locations.  Should run after
  // box-to-stack promotion since it is limited to propagating through
  // stack locations. Should run before aggregate lowering since that
  // splits up copy_addr.
  P.addCopyForwarding();

  // Split up opaque operations (copy_addr, retain_value, etc.).
  P.addLowerAggregateInstrs();

  // Split up operations on stack-allocated aggregates (struct, tuple).
  P.addSROA();

  // Re-run predictable memory optimizations, since previous optimization
  // passes sometimes expose oppotunities here.
  P.addPredictableMemoryOptimizations();

  // Promote stack allocations to values.
  P.addMem2Reg();

  // Cleanup, which is important if the inliner has restarted the pass pipeline.
  P.addPerformanceConstantPropagation();
  P.addSimplifyCFG();
  P.addSILCombine();

  // Mainly for Array.append(contentsOf) optimization.
  P.addArrayElementPropagation();
  
  // Run the devirtualizer, specializer, and inliner. If any of these
  // makes a change we'll end up restarting the function passes on the
  // current function (after optimizing any new callees).
  P.addDevirtualizer();
  P.addGenericSpecializer();
  // Run devirtualizer after the specializer, because many
  // class_method/witness_method instructions may use concrete types now.
  P.addDevirtualizer();

  switch (OpLevel) {
  case OptimizationLevelKind::HighLevel:
    // Does not inline functions with defined semantics.
    P.addEarlyInliner();
    break;
  case OptimizationLevelKind::MidLevel:
    P.addGlobalOpt();
    P.addLetPropertiesOpt();
    // It is important to serialize before any of the @_semantics
    // functions are inlined, because otherwise the information about
    // uses of such functions inside the module is lost,
    // which reduces the ability of the compiler to optimize clients
    // importing this module.
    P.addSerializeSILPass();
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
  P.addSILCombine();
  // SILCombine can expose further opportunities for SimplifyCFG.
  P.addSimplifyCFG();

  P.addCSE();
  P.addRedundantLoadElimination();

  P.addPerformanceConstantPropagation();
  P.addCSE();
  P.addDCE();

  // Perform retain/release code motion and run the first ARC optimizer.
  P.addEarlyCodeMotion();
  P.addReleaseHoisting();
  P.addARCSequenceOpts();

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
  P.addRemovePins();
}
