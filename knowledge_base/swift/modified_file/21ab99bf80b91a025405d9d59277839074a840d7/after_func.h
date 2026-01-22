  llvm::SmallSetVector<SILInstruction *, 1>
  computeEpilogueARCInstructions(EpilogueARCContext::EpilogueARCKind Kind,
                                 SILValue Arg) {
    auto &ARCCache = Kind == EpilogueARCContext::EpilogueARCKind::Retain ?
                 EpilogueRetainInstCache :
                 EpilogueReleaseInstCache;
    auto Iter = ARCCache.find(Arg);
    if (Iter != ARCCache.end())
      return Iter->second;

    EpilogueARCContext CM(Kind, Arg, F, PO->get(F), AA, RC->get(F));
    // Initialize and run the data flow. Clear the epilogue arc instructions if the
    // data flow is aborted in middle.
   if (!CM.run()) { 
     CM.resetEpilogueARCInsts();
     return CM.getEpilogueARCInsts();
    }
    return ARCCache[Arg] = CM.getEpilogueARCInsts();
  }
