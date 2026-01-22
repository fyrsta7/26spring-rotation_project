SILInstruction *
CastOptimizer::
optimizeUnconditionalCheckedCastAddrInst(UnconditionalCheckedCastAddrInst *Inst) {
  auto Loc = Inst->getLoc();
  auto Src = Inst->getSrc();
  auto Dest = Inst->getDest();
  auto SourceType = Inst->getSourceType();
  auto TargetType = Inst->getTargetType();
  auto &Mod = Inst->getModule();

  bool isSourceTypeExact = isa<MetatypeInst>(Src);

  // Check if we can statically predict the outcome of the cast.
  auto Feasibility = classifyDynamicCast(Mod.getSwiftModule(), SourceType,
                                         TargetType, isSourceTypeExact);

  if (Feasibility == DynamicCastFeasibility::MaySucceed) {
    return nullptr;
  }

  if (Feasibility == DynamicCastFeasibility::WillFail) {
    // Remove the cast and insert a trap, followed by an
    // unreachable instruction.
    SILBuilderWithScope<1> Builder(Inst);
    SILInstruction *NewI = Builder.createBuiltinTrap(Loc);
    // mem2reg's invariants get unhappy if we don't try to
    // initialize a loadable result.
    auto DestType = Dest.getType();
    auto &resultTL = Mod.Types.getTypeLowering(DestType);
    if (!resultTL.isAddressOnly()) {
      auto undef = SILValue(SILUndef::get(DestType.getObjectType(),
                                          Builder.getModule()));
      NewI = Builder.createStore(Loc, undef, Dest);
    }
    Inst->replaceAllUsesWithUndef();
    EraseInstAction(Inst);
    Builder.setInsertionPoint(std::next(SILBasicBlock::iterator(NewI)));
    Builder.createUnreachable(ArtificialUnreachableLocation());
    WillFailAction();
  }

  if (Feasibility == DynamicCastFeasibility::WillSucceed) {

    bool ResultNotUsed = isa<AllocStackInst>(Dest.getDef());
    for (auto Use : Dest.getUses()) {
      auto *User = Use->getUser();
      if (isa<DeallocStackInst>(User) || User == Inst)
        continue;
      ResultNotUsed = false;
      break;
    }

    if (ResultNotUsed) {
      EraseInstAction(Inst);
      WillSucceedAction();
      return nullptr;
    }

    // Try to apply the bridged casts optimizations
    auto NewI = optimizeBridgedCasts(Inst, false, Src, Dest, SourceType,
                                         TargetType, nullptr, nullptr);
    if (NewI) {
        WillSucceedAction();
        return nullptr;
    }

    if (isBridgingCast(SourceType, TargetType))
      return nullptr;

    SILBuilderWithScope<1> Builder(Inst);
    if (!emitSuccessfulIndirectUnconditionalCast(Builder, Mod.getSwiftModule(),
                                            Loc, Inst->getConsumptionKind(),
                                            Src, SourceType,
                                            Dest, TargetType, Inst)) {
      // No optimization was possible.
      return nullptr;
    }

    Inst->replaceAllUsesWithUndef();
    EraseInstAction(Inst);
    WillSucceedAction();
  }

  return nullptr;
}
