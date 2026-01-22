TypeSubstitutionMap
TypeBase::getContextSubstitutions(const DeclContext *dc,
                                  GenericEnvironment *genericEnv) {
  assert(dc->isTypeContext());
  Type baseTy(this);

  assert(!baseTy->hasLValueType() &&
         !baseTy->is<AnyMetatypeType>() &&
         !baseTy->is<ErrorType>());

  // The resulting set of substitutions. Always use this to ensure we
  // don't miss out on NRVO anywhere.
  TypeSubstitutionMap substitutions;

  // If the member is part of a protocol or extension thereof, we need
  // to substitute in the type of Self.
  if (dc->getSelfProtocolDecl()) {
    // FIXME: This feels painfully inefficient. We're creating a dense map
    // for a single substitution.
    substitutions[dc->getSelfInterfaceType()
                    ->getCanonicalType()->castTo<GenericTypeParamType>()]
      = baseTy;
    return substitutions;
  }

  const auto genericSig = dc->getGenericSignatureOfContext();
  if (!genericSig)
    return substitutions;

  // Find the superclass type with the context matching that of the member.
  auto *ownerNominal = dc->getSelfNominalTypeDecl();
  if (auto *ownerClass = dyn_cast<ClassDecl>(ownerNominal))
    baseTy = baseTy->getSuperclassForDecl(ownerClass);

  // Gather all of the substitutions for all levels of generic arguments.
  auto params = genericSig->getGenericParams();
  unsigned n = params.size();

  while (baseTy && n > 0) {
    if (baseTy->is<ErrorType>())
      break;

    // For a bound generic type, gather the generic parameter -> generic
    // argument substitutions.
    if (auto boundGeneric = baseTy->getAs<BoundGenericType>()) {
      auto args = boundGeneric->getGenericArgs();
      for (unsigned i = 0, e = args.size(); i < e; ++i) {
        substitutions[params[n - e + i]->getCanonicalType()
                        ->castTo<GenericTypeParamType>()] = args[i];
      }

      // Continue looking into the parent.
      baseTy = boundGeneric->getParent();
      n -= args.size();
      continue;
    }

    // Continue looking into the parent.
    if (auto protocolTy = baseTy->getAs<ProtocolType>()) {
      baseTy = protocolTy->getParent();
      n--;
      continue;
    }

    // Continue looking into the parent.
    if (auto nominalTy = baseTy->getAs<NominalType>()) {
      baseTy = nominalTy->getParent();
      continue;
    }

    // Assert and break to avoid hanging if we get an unexpected baseTy.
    assert(0 && "Bad base type");
    break;
  }

  while (n > 0) {
    auto *gp = params[--n];
    auto substTy = (genericEnv
                    ? genericEnv->mapTypeIntoContext(gp)
                    : gp);
    auto result = substitutions.insert(
      {gp->getCanonicalType()->castTo<GenericTypeParamType>(),
       substTy});
    assert(result.second);
    (void) result;
  }

  return substitutions;
}
