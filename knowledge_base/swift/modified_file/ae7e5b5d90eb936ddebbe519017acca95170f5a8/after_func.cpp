void LoopTreeOptimization::analyzeCurrentLoop(
    std::unique_ptr<LoopNestSummary> &CurrSummary) {
  InstSet &sideEffects = CurrSummary->SideEffectInsts;
  SILLoop *Loop = CurrSummary->Loop;
  LLVM_DEBUG(llvm::dbgs() << " Analyzing accesses.\n");

  auto *Preheader = Loop->getLoopPreheader();
  if (!Preheader) {
    // Can't hoist/sink instructions
    return;
  }

  // Interesting instructions in the loop:
  SmallVector<ApplyInst *, 8> ReadOnlyApplies;
  SmallVector<ApplyInst *, 8> globalInitCalls;
  SmallVector<LoadInst *, 8> Loads;
  SmallVector<StoreInst *, 8> Stores;
  SmallVector<FixLifetimeInst *, 8> FixLifetimes;
  SmallVector<BeginAccessInst *, 8> BeginAccesses;
  SmallVector<FullApplySite, 8> fullApplies;

  for (auto *BB : Loop->getBlocks()) {
    SmallVector<SILInstruction *, 8> sideEffectsInBlock;
    for (auto &Inst : *BB) {
      switch (Inst.getKind()) {
      case SILInstructionKind::FixLifetimeInst: {
        auto *FL = cast<FixLifetimeInst>(&Inst);
        if (DomTree->dominates(FL->getOperand()->getParentBlock(), Preheader))
          FixLifetimes.push_back(FL);
        // We can ignore the side effects of FixLifetimes
        break;
      }
      case SILInstructionKind::LoadInst:
        Loads.push_back(cast<LoadInst>(&Inst));
        LoadsAndStores.push_back(&Inst);
        break;
      case SILInstructionKind::StoreInst: {
        Stores.push_back(cast<StoreInst>(&Inst));
        LoadsAndStores.push_back(&Inst);
        checkSideEffects(Inst, sideEffects, sideEffectsInBlock);
        break;
      }
      case SILInstructionKind::BeginAccessInst:
        BeginAccesses.push_back(cast<BeginAccessInst>(&Inst));
        checkSideEffects(Inst, sideEffects, sideEffectsInBlock);
        break;
      case SILInstructionKind::RefElementAddrInst:
        SpecialHoist.push_back(cast<RefElementAddrInst>(&Inst));
        break;
      case swift::SILInstructionKind::CondFailInst:
        // We can (and must) hoist cond_fail instructions if the operand is
        // invariant. We must hoist them so that we preserve memory safety. A
        // cond_fail that would have protected (executed before) a memory access
        // must - after hoisting - also be executed before said access.
        HoistUp.insert(&Inst);
        checkSideEffects(Inst, sideEffects, sideEffectsInBlock);
        break;
      case SILInstructionKind::ApplyInst: {
        auto *AI = cast<ApplyInst>(&Inst);
        if (isSafeReadOnlyApply(BCA, AI)) {
          ReadOnlyApplies.push_back(AI);
        } else if (SILFunction *callee = AI->getReferencedFunctionOrNull()) {
          // Calls to global inits are different because we don't care about
          // side effects which are "after" the call in the loop.
          if (callee->isGlobalInit() &&
              // Check against side-effects within the same block.
              // Side-effects in other blocks are checked later (after we
              // scanned all blocks of the loop).
              !mayConflictWithGlobalInit(AA, sideEffectsInBlock, AI))
            globalInitCalls.push_back(AI);
        }
        // check for array semantics and side effects - same as default
        LLVM_FALLTHROUGH;
      }
      default:
        if (auto fullApply = FullApplySite::isa(&Inst)) {
          fullApplies.push_back(fullApply);
        }
        checkSideEffects(Inst, sideEffects, sideEffectsInBlock);
        if (canHoistUpDefault(&Inst, Loop, DomTree, RunsOnHighLevelSIL)) {
          HoistUp.insert(&Inst);
        }
        break;
      }
    }
  }

  // Avoid quadratic complexity in corner cases. Usually, this limit will not be exceeded.
  if (ReadOnlyApplies.size() * sideEffects.size() < 8000) {
    for (auto *AI : ReadOnlyApplies) {
      if (!mayWriteTo(AA, BCA, sideEffects, AI)) {
        HoistUp.insert(AI);
      }
    }
  }
  // Avoid quadratic complexity in corner cases. Usually, this limit will not be exceeded.
  if (Loads.size() * sideEffects.size() < 8000) {
    for (auto *LI : Loads) {
      if (!mayWriteTo(AA, sideEffects, LI)) {
        HoistUp.insert(LI);
      }
    }
  }

  if (!globalInitCalls.empty()) {
    if (!postDomTree) {
      postDomTree = PDA->get(Preheader->getParent());
    }
    if (postDomTree->getRootNode()) {
      for (ApplyInst *ginitCall : globalInitCalls) {
        // Check against side effects which are "before" (i.e. post-dominated
        // by) the global initializer call.
        if (!mayConflictWithGlobalInit(AA, sideEffects, ginitCall, Preheader,
             postDomTree)) {
          HoistUp.insert(ginitCall);
        }
      }
    }
  }

  // Collect memory locations for which we can move all loads and stores out
  // of the loop.
  //
  // Note: The Loads set and LoadsAndStores set may mutate during this loop.
  for (StoreInst *SI : Stores) {
    // Use AccessPathWithBase to recover a base address that can be used for
    // newly inserted memory operations. If we instead teach hoistLoadsAndStores
    // how to rematerialize global_addr, then we don't need this base.
    auto access = AccessPathWithBase::compute(SI->getDest());
    auto accessPath = access.accessPath;
    if (accessPath.isValid() &&
        (access.base && isLoopInvariant(access.base, Loop))) {
      if (isOnlyLoadedAndStored(AA, sideEffects, Loads, Stores, SI->getDest(),
                                accessPath)) {
        if (!LoadAndStoreAddrs.count(accessPath)) {
          if (splitLoads(Loads, accessPath, SI->getDest())) {
            LoadAndStoreAddrs.insert(accessPath);
          }
        }
      }
    }
  }
  if (!FixLifetimes.empty()) {
    bool sideEffectsMayRelease =
        std::any_of(sideEffects.begin(), sideEffects.end(),
                    [&](SILInstruction *W) { return W->mayRelease(); });
    for (auto *FL : FixLifetimes) {
      if (!sideEffectsMayRelease || !mayWriteTo(AA, sideEffects, FL)) {
        SinkDown.push_back(FL);
      }
    }
  }
  for (auto *BI : BeginAccesses) {
    if (!handledEndAccesses(BI, Loop)) {
      LLVM_DEBUG(llvm::dbgs() << "Skipping: " << *BI);
      LLVM_DEBUG(llvm::dbgs() << "Some end accesses can't be handled\n");
      continue;
    }
    if (analyzeBeginAccess(BI, BeginAccesses, fullApplies, sideEffects, ASA,
                           DomTree)) {
      SpecialHoist.push_back(BI);
    }
  }
}
