GenericEnvironment *
TypeChecker::handleSILGenericParams(GenericParamList *genericParams,
                                    DeclContext *DC) {

  SmallVector<GenericParamList *, 2> nestedList;
  for (; genericParams; genericParams = genericParams->getOuterParameters()) {
    nestedList.push_back(genericParams);
  }

  std::reverse(nestedList.begin(), nestedList.end());

  // Since the innermost GenericParamList is in the beginning of the vector,
  // we process in reverse order to handle the outermost list first.
  GenericSignature *parentSig = nullptr;
  GenericEnvironment *parentEnv = nullptr;

  for (unsigned i = 0, e = nestedList.size(); i < e; ++i) {
    auto genericParams = nestedList[i];
    genericParams->setDepth(i);

    parentEnv = checkGenericEnvironment(genericParams, DC, parentSig,
                                        /*allowConcreteGenericParams=*/true,
                                        /*ext=*/nullptr);
    parentSig = parentEnv->getGenericSignature();
  }

  return parentEnv;
}
