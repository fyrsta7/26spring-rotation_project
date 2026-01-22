void swift::runSILOptimizationPasses(SILModule &Module,
                                     const SILOptions &Options) {
    SILPassManager PM(&Module, Options);
    PM.registerAnalysis(createCallGraphAnalysis(&Module));
    PM.registerAnalysis(createAliasAnalysis(&Module));
    PM.registerAnalysis(createDominanceAnalysis(&Module));
    PM.add(createSILLinker());
    if (Options.DebugSerialization) {
      PM.run();
      return;
    }
    PM.add(createGenericSpecializer());
    PM.add(createPerfInliner());
    PM.add(createSILCombine());
    PM.add(createDeadFunctionElimination());
    PM.add(createLowerAggregate());
    PM.add(createSROA());
    PM.add(createMem2Reg());
    PM.add(createPerformanceConstantPropagation());
    PM.add(createCSE());
    PM.add(createSILCombine());
    PM.add(createLoadStoreOpts());
    PM.add(createCodeMotion());
    PM.add(createSimplifyCFG());
    PM.add(createDevirtualization());
    PM.add(createARCOpts());
    PM.add(createAllocBoxToStack());
    PM.add(createDeadObjectElimination());
    PM.run();

    DEBUG(Module.verify());
}
