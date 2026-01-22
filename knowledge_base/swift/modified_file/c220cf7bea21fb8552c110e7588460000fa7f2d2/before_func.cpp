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

  // Just make sure that we do not have side effects.
  return Inst->getMemoryBehavior() !=
    SILInstruction::MemoryBehavior::MayHaveSideEffects;
}
