void AddSSAPasses(SILPassManager &PM, OptimizationLevelKind OpLevel) {
  AddSimplifyCFGSILCombine(PM);
  PM.addAllocBoxToStack();
  PM.addCopyForwarding();
  PM.addLowerAggregateInstrs();
  PM.addSILCombine();
  PM.addSROA();
  PM.addMem2Reg();

  // Perform classic SSA optimizations.
  PM.addGlobalOpt();
  PM.addLetPropertiesOpt();
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
  PM.addRedundantLoadElimination();
  PM.addDeadStoreElimination();
  PM.addCSE();
  PM.addEarlyCodeMotion();
  PM.addARCSequenceOpts();

  PM.addSILLinker();

  switch (OpLevel) {
    case OptimizationLevelKind::HighLevel:
      // Does not inline functions with defined semantics.
      PM.addEarlyInliner();
      break;
    case OptimizationLevelKind::MidLevel:
      // Does inline semantics-functions (except "availability"), but not
      // global-init functions.
      PM.addPerfInliner();
      break;
    case OptimizationLevelKind::LowLevel:
      // Inlines everything
      PM.addLateInliner();
      break;
  }
  PM.addSimplifyCFG();
  // Only hoist releases very late.
  if (OpLevel == OptimizationLevelKind::LowLevel)
    PM.addLateCodeMotion();
  else
    PM.addEarlyCodeMotion();
  PM.addARCSequenceOpts();
  PM.addRemovePins();
}
