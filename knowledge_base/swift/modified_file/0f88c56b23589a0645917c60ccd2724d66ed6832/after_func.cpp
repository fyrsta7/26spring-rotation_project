SILInstruction *
CastOptimizer::
optimizeUnconditionalCheckedCastInst(UnconditionalCheckedCastInst *Inst) {
  auto LoweredSourceType = Inst->getOperand().getType();
  auto LoweredTargetType = Inst->getType();
  auto Loc = Inst->getLoc();
  auto Op = Inst->getOperand();
  auto &Mod = Inst->getModule();

  bool isSourceTypeExact = isa<MetatypeInst>(Op);

  // Check if we can statically predict the outcome of the cast.
  auto Feasibility = classifyDynamicCast(Mod.getSwiftModule(),
                          LoweredSourceType.getSwiftRValueType(),
                          LoweredTargetType.getSwiftRValueType(),
                          isSourceTypeExact);

  if (Feasibility == DynamicCastFeasibility::MaySucceed) {
    return nullptr;
  }

  if (Feasibility == DynamicCastFeasibility::WillFail) {
    // Remove the cast and insert a trap, followed by an
    // unreachable instruction.
    SILBuilderWithScope<1> Builder(Inst);
    auto *Trap = Builder.createBuiltinTrap(Loc);
    Inst->replaceAllUsesWithUndef();
    EraseInstAction(Inst);
    Builder.setInsertionPoint(std::next(SILBasicBlock::iterator(Trap)));
    Builder.createUnreachable(ArtificialUnreachableLocation());
    WillFailAction();
    return Trap;
  }

  if (Feasibility == DynamicCastFeasibility::WillSucceed) {

    if (Inst->use_empty()) {
      EraseInstAction(Inst);
      WillSucceedAction();
      return nullptr;
    }

    SILBuilderWithScope<1> Builder(Inst);

    // Try to apply the bridged casts optimizations
    auto SourceType = LoweredSourceType.getSwiftRValueType();
    auto TargetType = LoweredTargetType.getSwiftRValueType();
    auto Src = Inst->getOperand();
    auto NewI = optimizeBridgedCasts(Inst, false, Src, SILValue(), SourceType,
        TargetType, nullptr, nullptr);
    if (NewI) {
      ReplaceInstUsesAction(Inst, NewI);
      EraseInstAction(Inst);
      WillSucceedAction();
      return NewI;
    }

    if (isBridgingCast(SourceType, TargetType))
      return nullptr;

    auto Result = emitSuccessfulScalarUnconditionalCast(Builder,
                      Mod.getSwiftModule(), Loc, Op,
                      LoweredTargetType,
                      LoweredSourceType.getSwiftRValueType(),
                      LoweredTargetType.getSwiftRValueType(),
                      Inst);

    if (!Result) {
      // No optimization was possible.
      return nullptr;
    }

    ReplaceInstUsesAction(Inst, Result.getDef());
    EraseInstAction(Inst);
    WillSucceedAction();
    return dyn_cast<SILInstruction>(Result.getDef());
  }

  return nullptr;
}
