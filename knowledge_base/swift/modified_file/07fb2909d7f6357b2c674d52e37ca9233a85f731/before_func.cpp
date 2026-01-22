void AddSSAPasses(SILPassManager &PM, OptimizationLevelKind OpLevel) {
  // Promote box allocations to stack allocations.
  PM.addAllocBoxToStack();

  // Propagate copies through stack locations.  Should run after
  // box-to-stack promotion since it is limited to propagating through
  // stack locations. Should run before aggregate lowering since that
  // splits up copy_addr.
  PM.addCopyForwarding();

  // Split up opaque operations (copy_addr, retain_value, etc.).
  PM.addLowerAggregateInstrs();

  // Split up operations on stack-allocated aggregates (struct, tuple).
  PM.addSROA();

  // Promote stack allocations to values.
  PM.addMem2Reg();

  // Run the devirtualizer, specializer, and inliner. If any of these
  // makes a change we'll end up restarting the function passes on the
  // current function (after optimizing any new callees).
  PM.addDevirtualizer();
  PM.addGenericSpecializer();

  switch (OpLevel) {
    case OptimizationLevelKind::HighLevel:
      // Does not inline functions with defined semantics.
      PM.addEarlyInliner();
      break;
    case OptimizationLevelKind::MidLevel:
      // Does inline semantics-functions (except "availability"), but not
      // global-init functions.
      PM.addGlobalOpt();
      PM.addLetPropertiesOpt();
      PM.addPerfInliner();
      break;
    case OptimizationLevelKind::LowLevel:
      // Inlines everything
      PM.addLateInliner();
      break;
  }

  // Promote stack allocations to values and eliminate redundant
  // loads.
  PM.addMem2Reg();
  PM.addRedundantLoadElimination();
  //  Do a round of CFG simplification, followed by peepholes, then
  //  more CFG simplification.
  AddSimplifyCFGSILCombine(PM);

  PM.addPerformanceConstantPropagation();
  PM.addDCE();
  PM.addCSE();
  PM.addSILCombine();
  PM.addJumpThreadSimplifyCFG();
  // Jump threading can expose opportunity for silcombine (enum -> is_enum_tag->
  // cond_br).
  PM.addSILCombine();
  // Which can expose opportunity for simplifcfg.
  PM.addSimplifyCFG();

  // Perform retain/release code motion and run the first ARC optimizer.
  PM.addCSE();
  PM.addEarlyCodeMotion();
  PM.addARCSequenceOpts();

  PM.addSimplifyCFG();
  if (OpLevel == OptimizationLevelKind::LowLevel) {
    // Remove retain/releases based on Builtin.unsafeGuaranteed
    PM.addUnsafeGuaranteedPeephole();
    // Only hoist releases very late.
    PM.addLateCodeMotion();
  } else
    PM.addEarlyCodeMotion();

  PM.addARCSequenceOpts();
  PM.addRemovePins();
}
