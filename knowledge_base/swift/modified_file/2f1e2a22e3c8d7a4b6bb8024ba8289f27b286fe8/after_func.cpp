SILInstruction *SILCombiner::visitCopyValueInst(CopyValueInst *CI) {
  SILValue Operand = CI->getOperand();
  SILType OperandTy = Operand.getType();

  // copy_value of an enum with a trivial payload or no-payload is a no-op +
  // RAUW.
  if (auto *EI = dyn_cast<EnumInst>(Operand.getDef()))
    if (!EI->hasOperand() || EI->getOperand().getType().isTrivial(Module)) {
      // We need to use eraseInstFromFunction + RAUW here since a copy value can
      // never be trivially dead since it touches reference counts.      
      replaceInstUsesWith(*CI, EI, 0);
      return eraseInstFromFunction(*CI);
    }

  // CopyValueInst of a reference type is a strong_release.
  if (OperandTy.hasReferenceSemantics()) {
    Builder->createStrongRetain(CI->getLoc(), Operand);
    // We need to use eraseInstFromFunction + RAUW here since a copy value can
    // never be trivially dead since it touches reference counts.
    replaceInstUsesWith(*CI, Operand.getDef(), 0);
    return eraseInstFromFunction(*CI);
  }

  // CopyValueInst of a trivial type is a no-op + use propogation.
  if (OperandTy.isTrivial(Module)) {
    // We need to use eraseInstFromFunction + RAUW here since a copy value can
    // never be trivially dead since it touches reference counts.
    replaceInstUsesWith(*CI, Operand.getDef(), 0);
    return eraseInstFromFunction(*CI);
  }

  // Do nothing for non-trivial non-reference types.
  return nullptr;
}
