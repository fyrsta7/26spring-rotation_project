bool swift::arc::cannotDecrementRefCount(SILInstruction *Inst,
                                         SILValue Target) {
  // Clear up some small cases where MayHaveSideEffects is too broad for our
  // purposes and the instruction does not decrement ref counts.
  switch (Inst->getKind()) {
  case ValueKind::DeallocStackInst:
  case ValueKind::StrongRetainInst:
  case ValueKind::StrongRetainAutoreleasedInst:
  case ValueKind::StrongRetainUnownedInst:
  case ValueKind::UnownedRetainInst:
  case ValueKind::PartialApplyInst:
  case ValueKind::CondFailInst:
    return true;
  case ValueKind::CopyAddrInst: {
    auto *CA = cast<CopyAddrInst>(Inst);
    if (CA->isInitializationOfDest() == IsInitialization_t::IsInitialization)
      return true;
  }
  SWIFT_FALLTHROUGH;
  default:
    break;
  }

  if (auto *AI = dyn_cast<ApplyInst>(Inst)) {
    // Ignore any thick functions for now due to us not handling the ref-counted
    // nature of its context.
    if (auto FTy = AI->getCallee().getType().getAs<SILFunctionType>())
      if (!FTy->isThin())
        return false;

    // If we have a builtin that is side effect free, we can commute the
    // ApplyInst and the retain.
    if (auto *BI = dyn_cast<BuiltinFunctionRefInst>(AI->getCallee()))
      if (isSideEffectFree(BI))
        return true;

    return false;
  }

  // Just make sure that we do not have side effects.
  return Inst->getMemoryBehavior() !=
    SILInstruction::MemoryBehavior::MayHaveSideEffects;
}
