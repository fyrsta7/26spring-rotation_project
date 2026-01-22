  void run() {
    SILFunction &F = *getFunction();

    DEBUG(llvm::dbgs() << "***** Load Store Elimination on function: "
          << F.getName() << " *****\n");

    AliasAnalysis *AA = PM->getAnalysis<AliasAnalysis>();

    // Remove dead stores, merge duplicate loads, and forward stores to loads.
    bool Changed = false;
    for (auto &BB : F)
      Changed |= performLoadStoreOptimizations(&BB, AA);

    if (Changed)
      invalidateAnalysis(SILAnalysis::InvalidationKind::Instructions);
  }
