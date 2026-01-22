bool SILPerformanceInliner::isProfitableToInline(
    FullApplySite AI, Weight CallerWeight, ConstantTracker &callerTracker,
    int &NumCallerBlocks,
    const llvm::DenseMap<SILBasicBlock *, uint64_t> &BBToWeightMap) {
  SILFunction *Callee = AI.getReferencedFunction();
  bool IsGeneric = AI.hasSubstitutions();

  assert(EnableSILInliningOfGenerics || !IsGeneric);

  // Start with a base benefit.
  int BaseBenefit = RemovedCallBenefit;

  // Osize heuristic.
  bool isClassMethodAtOsize = false;
  if (OptMode == OptimizationMode::ForSize) {
    // Don't inline into thunks.
    if (AI.getFunction()->isThunk())
      return false;

    // Don't inline class methods.
    if (Callee->hasSelfParam()) {
      auto SelfTy = Callee->getLoweredFunctionType()->getSelfInstanceType();
      if (SelfTy->mayHaveSuperclass() &&
          Callee->getRepresentation() == SILFunctionTypeRepresentation::Method)
        isClassMethodAtOsize = true;
    }
    // Use command line option to control inlining in Osize mode.
    const uint64_t CallerBaseBenefitReductionFactor = AI.getFunction()->getModule().getOptions().CallerBaseBenefitReductionFactor;
    BaseBenefit = BaseBenefit / CallerBaseBenefitReductionFactor;
  }

  // It is always OK to inline a simple call.
  // TODO: May be consider also the size of the callee?
  if (isPureCall(AI, SEA)) {
    LLVM_DEBUG(dumpCaller(AI.getFunction());
               llvm::dbgs() << "    pure-call decision " << Callee->getName()
                            << '\n');
    return true;
  }

  // Bail out if this generic call can be optimized by means of
  // the generic specialization, because we prefer generic specialization
  // to inlining of generics.
  if (IsGeneric && canSpecializeGeneric(AI, Callee, AI.getSubstitutionMap())) {
    return false;
  }

  SILLoopInfo *LI = LA->get(Callee);
  ShortestPathAnalysis *SPA = getSPA(Callee, LI);
  assert(SPA->isValid());

  ConstantTracker constTracker(Callee, &callerTracker, AI);
  DominanceInfo *DT = DA->get(Callee);
  SILBasicBlock *CalleeEntry = &Callee->front();
  DominanceOrder domOrder(CalleeEntry, DT, Callee->size());

  // We don't want to blow up code-size
  // We will only inline if *ALL* dynamic accesses are
  // known and have no nested conflict
  bool AllAccessesBeneficialToInline = true;

  // Calculate the inlining cost of the callee.
  int CalleeCost = 0;
  int Benefit = 0;
  // We don’t know if we want to update the benefit with
  // the exclusivity heuristic or not. We can *only* do that
  // if AllAccessesBeneficialToInline is true
  int ExclusivityBenefitWeight = 0;

  SubstitutionMap CalleeSubstMap = AI.getSubstitutionMap();

  CallerWeight.updateBenefit(Benefit, BaseBenefit);

  // Go through all blocks of the function, accumulate the cost and find
  // benefits.
  while (SILBasicBlock *block = domOrder.getNext()) {
    constTracker.beginBlock();
    Weight BlockW = SPA->getWeight(block, CallerWeight);

    for (SILInstruction &I : *block) {
      constTracker.trackInst(&I);

      CalleeCost += (int)instructionInlineCost(I);

      if (FullApplySite FAI = FullApplySite::isa(&I)) {
        // Check if the callee is passed as an argument. If so, increase the
        // threshold, because inlining will (probably) eliminate the closure.
        SILInstruction *def = constTracker.getDefInCaller(FAI.getCallee());
        if (def && (isa<FunctionRefInst>(def) || isa<PartialApplyInst>(def)))
          BlockW.updateBenefit(Benefit, RemovedClosureBenefit);
        // Check if inlining the callee would allow for further
        // optimizations like devirtualization or generic specialization. 
        if (!def)
          def = dyn_cast_or_null<SingleValueInstruction>(FAI.getCallee());

        if (!def)
          continue;

        auto Subs = FAI.getSubstitutionMap();

        // Bail if it is not a generic call or inlining of generics is forbidden.
        if (!EnableSILInliningOfGenerics || Subs.empty())
          continue;

        if (!isa<FunctionRefInst>(def) && !isa<ClassMethodInst>(def) &&
            !isa<WitnessMethodInst>(def))
          continue;

        // It is a generic call inside the callee. Check if after inlining
        // it will be possible to perform a generic specialization or
        // devirtualization of this call.

        // Create the list of substitutions as they will be after
        // inlining.
        auto SubMap = Subs.subst(CalleeSubstMap);

        // Check if the call can be devirtualized.
        if (isa<ClassMethodInst>(def) || isa<WitnessMethodInst>(def) ||
            isa<SuperMethodInst>(def)) {
          // TODO: Take AI.getSubstitutions() into account.
          if (canDevirtualizeApply(FAI, nullptr)) {
            LLVM_DEBUG(llvm::dbgs() << "Devirtualization will be possible "
                                       "after inlining for the call:\n";
                       FAI.getInstruction()->dumpInContext());
            BlockW.updateBenefit(Benefit, DevirtualizedCallBenefit);
          }
        }

        // Check if a generic specialization would be possible.
        if (isa<FunctionRefInst>(def)) {
          auto CalleeF = FAI.getCalleeFunction();
          if (!canSpecializeGeneric(FAI, CalleeF, SubMap))
            continue;
          LLVM_DEBUG(llvm::dbgs() << "Generic specialization will be possible "
                                     "after inlining for the call:\n";
                     FAI.getInstruction()->dumpInContext());
          BlockW.updateBenefit(Benefit, GenericSpecializationBenefit);
        }
      } else if (auto *LI = dyn_cast<LoadInst>(&I)) {
        // Check if it's a load from a stack location in the caller. Such a load
        // might be optimized away if inlined.
        if (constTracker.isStackAddrInCaller(LI->getOperand()))
          BlockW.updateBenefit(Benefit, RemovedLoadBenefit);
      } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
        // Check if it's a store to a stack location in the caller. Such a load
        // might be optimized away if inlined.
        if (constTracker.isStackAddrInCaller(SI->getDest()))
          BlockW.updateBenefit(Benefit, RemovedStoreBenefit);
      } else if (isa<StrongReleaseInst>(&I) || isa<ReleaseValueInst>(&I)) {
        SILValue Op = stripCasts(I.getOperand(0));
        if (auto *Arg = dyn_cast<SILFunctionArgument>(Op)) {
          if (Arg->getArgumentConvention() ==
              SILArgumentConvention::Direct_Guaranteed) {
            BlockW.updateBenefit(Benefit, RefCountBenefit);
          }
        }
      } else if (auto *BI = dyn_cast<BuiltinInst>(&I)) {
        if (BI->getBuiltinInfo().ID == BuiltinValueKind::OnFastPath)
          BlockW.updateBenefit(Benefit, FastPathBuiltinBenefit);
      } else if (auto *BAI = dyn_cast<BeginAccessInst>(&I)) {
        if (BAI->getEnforcement() == SILAccessEnforcement::Dynamic) {
          // The access is dynamic and has no nested conflict
          // See if the storage location is considered by
          // access enforcement optimizations
          AccessedStorage storage =
              findAccessedStorageNonNested(BAI->getSource());
          if (BAI->hasNoNestedConflict() &&
              (storage.isUniquelyIdentified() ||
               storage.getKind() == AccessedStorage::Class)) {
            BlockW.updateBenefit(ExclusivityBenefitWeight, ExclusivityBenefit);
          } else {
            AllAccessesBeneficialToInline = false;
          }
        }
      }
    }
    // Don't count costs in blocks which are dead after inlining.
    SILBasicBlock *takenBlock = constTracker.getTakenBlock(block->getTerminator());
    if (takenBlock) {
      BlockW.updateBenefit(Benefit, RemovedTerminatorBenefit);
      domOrder.pushChildrenIf(block, [=](SILBasicBlock *child) {
        return child->getSinglePredecessorBlock() != block ||
               child == takenBlock;
      });
    } else {
      domOrder.pushChildren(block);
    }
  }

  if (AllAccessesBeneficialToInline) {
    Benefit = std::max(Benefit, ExclusivityBenefitWeight);
  }

  if (AI.getFunction()->isThunk()) {
    // Only inline trivial functions into thunks (which will not increase the
    // code size).
    if (CalleeCost > TrivialFunctionThreshold) {
      return false;
    }

    LLVM_DEBUG(dumpCaller(AI.getFunction());
               llvm::dbgs() << "    decision {" << CalleeCost << " into thunk} "
                            << Callee->getName() << '\n');
    return true;
  }

  // We reduce the benefit if the caller is too large. For this we use a
  // cubic function on the number of caller blocks. This starts to prevent
  // inlining at about 800 - 1000 caller blocks.
  if (NumCallerBlocks < BlockLimitMaxIntNumerator)
    Benefit -= 
      (NumCallerBlocks * NumCallerBlocks) / BlockLimitDenominator *
                          NumCallerBlocks / BlockLimitDenominator;
  else
    // The calculation in the if branch would overflow if we performed it.
    Benefit = 0;

  // If we have profile info - use it for final inlining decision.
  auto *bb = AI.getInstruction()->getParent();
  auto bbIt = BBToWeightMap.find(bb);
  if (bbIt != BBToWeightMap.end()) {
    return profileBasedDecision(AI, Benefit, Callee, CalleeCost,
                                NumCallerBlocks, bbIt);
  }
  if (isClassMethodAtOsize && Benefit > OSizeClassMethodBenefit) {
    Benefit = OSizeClassMethodBenefit;
  }

  // This is the final inlining decision.
  if (CalleeCost > Benefit) {
    ORE.emit([&]() {
      using namespace OptRemark;
      return RemarkMissed("NoInlinedCost", *AI.getInstruction())
             << "Not profitable to inline function " << NV("Callee", Callee)
             << " (cost = " << NV("Cost", CalleeCost)
             << ", benefit = " << NV("Benefit", Benefit) << ")";
    });
    return false;
  }

  NumCallerBlocks += Callee->size();

  LLVM_DEBUG(dumpCaller(AI.getFunction());
             llvm::dbgs() << "    decision {c=" << CalleeCost
                          << ", b=" << Benefit
                          << ", l=" << SPA->getScopeLength(CalleeEntry, 0)
                          << ", c-w=" << CallerWeight
                          << ", bb=" << Callee->size()
                          << ", c-bb=" << NumCallerBlocks
                          << "} " << Callee->getName() << '\n');
  ORE.emit([&]() {
    using namespace OptRemark;
    return RemarkPassed("Inlined", *AI.getInstruction())
           << NV("Callee", Callee) << " inlined into "
           << NV("Caller", AI.getFunction())
           << " (cost = " << NV("Cost", CalleeCost)
           << ", benefit = " << NV("Benefit", Benefit) << ")";
  });

  return true;
}
