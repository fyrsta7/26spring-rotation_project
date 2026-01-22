GenericEnvironment *
TypeChecker::handleSILGenericParams(GenericParamList *genericParams,
                                    DeclContext *DC) {
  if (genericParams == nullptr)
    return nullptr;

  SmallVector<GenericParamList *, 2> nestedList;
  for (; genericParams; genericParams = genericParams->getOuterParameters()) {
    nestedList.push_back(genericParams);
  }

  std::reverse(nestedList.begin(), nestedList.end());

  for (unsigned i = 0, e = nestedList.size(); i < e; ++i) {
    auto genericParams = nestedList[i];
    genericParams->setDepth(i);
  }

  return checkGenericEnvironment(nestedList.back(), DC,
                                 /*parentSig=*/nullptr,
                                 /*allowConcreteGenericParams=*/true,
                                 /*ext=*/nullptr);
}
