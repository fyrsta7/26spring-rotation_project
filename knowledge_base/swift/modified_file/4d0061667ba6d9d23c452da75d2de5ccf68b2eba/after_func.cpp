static Type transformTypeForTypeLowering(ASTContext &context, Type type) {
  if (type->is<ReferenceStorageType>())
    return type;

  return type.transform(context, [&](Type type) -> Type {
    if (auto *ft = type->getAs<FunctionType>()) {
      Type inputType = transformTypeForTypeLowering(context, ft->getInput());
      Type resultType = transformTypeForTypeLowering(context, ft->getResult());
      if (inputType.getPointer() == ft->getInput().getPointer() &&
          resultType.getPointer() == ft->getResult().getPointer() &&
          ft->getExtInfo().isAutoClosure() == false)
        return type;
      return FunctionType::get(inputType, resultType,
                               ft->getExtInfo()
                                 .withIsAutoClosure(false),
                               ft->getASTContext());
    }

    if (auto *pft = type->getAs<PolymorphicFunctionType>()) {
      Type inputType = transformTypeForTypeLowering(context, pft->getInput());
      Type resultType = transformTypeForTypeLowering(context, pft->getResult());
      if (inputType.getPointer() == pft->getInput().getPointer() &&
          resultType.getPointer() == pft->getResult().getPointer() &&
          pft->getExtInfo().isAutoClosure() == false)
        return type;
      return PolymorphicFunctionType::get(inputType, resultType,
                                          &pft->getGenericParams(),
                                          pft->getExtInfo()
                                            .withIsAutoClosure(false),
                                          pft->getASTContext());
    }

    return type;
  });
}
