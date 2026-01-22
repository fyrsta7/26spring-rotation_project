  void run() {
    SILFunction &F = *getFunction();

    DEBUG(llvm::dbgs() << "***** Load Store Elimination on function: "
          << F.getName() << " *****\n");

    AliasAnalysis *AA = PM->getAnalysis<AliasAnalysis>();

    // Remove dead stores, merge duplicate loads, and forward stores to loads.
    bool Changed = false;
    for (auto &BB : F)
      while (performLoadStoreOptimizations(&BB, AA))
        Changed = true;

    if (Changed)
      invalidateAnalysis(SILAnalysis::InvalidationKind::Instructions);
  }
