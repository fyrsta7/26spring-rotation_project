  llvm::DITypeRefArray createParameterTypes(CanSILFunctionType FnTy) {
    SmallVector<llvm::Metadata *, 16> Parameters;

    GenericContextScope scope(IGM, FnTy->getInvocationGenericSignature());

    // The function return type is the first element in the list.
    createParameterType(Parameters, getResultTypeForDebugInfo(IGM, FnTy));

    for (auto Param : FnTy->getParameters())
      createParameterType(
          Parameters, IGM.silConv.getSILType(
                          Param, FnTy, IGM.getMaximalTypeExpansionContext()));

    return DBuilder.getOrCreateTypeArray(Parameters);
  }