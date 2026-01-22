void swift::runSILOptimizationPasses(SILModule &Module) {
  performSILMem2Reg(&Module);
  performSILCSE(&Module);
  performSILCombine(&Module);
  performSimplifyCFG(&Module);
  performSILSpecialization(&Module);
  performSILPerformanceInlining(&Module);
}
