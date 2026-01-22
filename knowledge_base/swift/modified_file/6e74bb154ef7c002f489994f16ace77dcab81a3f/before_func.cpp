void swift::runSILOptimizationPasses(SILModule &Module) {
  //performSILPerformanceInlining(&Module);
  performSILMem2Reg(&Module);
  performSILCSE(&Module);
  performSILCombine(&Module);
  performSimplifyCFG(&Module);
  performSILSpecialization(&Module);
  performSILPerformanceInlining(&Module);
}
