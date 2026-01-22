static Type transformTypeForTypeLowering(ASTContext &context, Type type) {
  if (type->is<ReferenceStorageType>())
    return type;

  return type.transform(context, [&](Type type) -> Type {
    if (auto *ft = type->getAs<FunctionType>()) {
      Type inputType = transformTypeForTypeLowering(context, ft->getInput());
      Type resultType = transformTypeForTypeLowering(context, ft->getResult());
      return FunctionType::get(inputType, resultType,
                               ft->getExtInfo()
                                 .withIsAutoClosure(false),
                               ft->getASTContext());
    }

    if (auto *pft = type->getAs<PolymorphicFunctionType>()) {
      Type inputType = transformTypeForTypeLowering(context, pft->getInput());
      Type resultType = transformTypeForTypeLowering(context, pft->getResult());
      return PolymorphicFunctionType::get(inputType, resultType,
                                          &pft->getGenericParams(),
                                          pft->getExtInfo()
                                            .withIsAutoClosure(false),
                                          pft->getASTContext());
    }

    return type;
  });
}
