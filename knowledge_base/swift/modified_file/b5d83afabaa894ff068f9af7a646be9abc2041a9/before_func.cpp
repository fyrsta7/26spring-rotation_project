  llvm::DITypeRefArray createParameterTypes(SILType SILTy) {
    if (!SILTy)
      return nullptr;
    return createParameterTypes(SILTy.castTo<SILFunctionType>());
  }
