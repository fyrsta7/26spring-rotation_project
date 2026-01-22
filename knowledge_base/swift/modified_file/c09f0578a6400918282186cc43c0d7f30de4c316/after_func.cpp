Optional<ProtocolConformanceRef>
SubstitutionMap::lookupConformance(CanType type, ProtocolDecl *proto) const {
  if (empty()) return None;

  // If we have an archetype, map out of the context so we can compute a
  // conformance access path.
  if (auto archetype = dyn_cast<ArchetypeType>(type)) {
    type = archetype->getInterfaceType()->getCanonicalType();
  }

  // Error path: if we don't have a type parameter, there is no conformance.
  // FIXME: Query concrete conformances in the generic signature?
  if (!type->isTypeParameter())
    return None;

  auto genericSig = getGenericSignature();

  // Fast path
  unsigned index = 0;
  for (auto reqt : genericSig->getRequirements()) {
    if (reqt.getKind() == RequirementKind::Conformance) {
      if (reqt.getFirstType()->isEqual(type) &&
          reqt.getSecondType()->isEqual(proto->getDeclaredType()))
        return getConformances()[index];

      index++;
    }
  }

  // Retrieve the starting conformance from the conformance map.
  auto getInitialConformance =
    [&](Type type, ProtocolDecl *proto) -> Optional<ProtocolConformanceRef> {
      unsigned conformanceIndex = 0;
      for (const auto &req : getGenericSignature()->getRequirements()) {
        if (req.getKind() != RequirementKind::Conformance)
          continue;

        // Is this the conformance we're looking for?
        if (req.getFirstType()->isEqual(type) &&
            req.getSecondType()->castTo<ProtocolType>()->getDecl() == proto) {
          return getConformances()[conformanceIndex];
        }

        ++conformanceIndex;
      }

      return None;
    };

  // If the type doesn't conform to this protocol, the result isn't formed
  // from these requirements.
  if (!genericSig->conformsToProtocol(type, proto)) {
    // Check whether the superclass conforms.
    if (auto superclass = genericSig->getSuperclassBound(type)) {
      return LookUpConformanceInSignature(*getGenericSignature())(
                                                 type->getCanonicalType(),
                                                 superclass,
                                                 proto);
    }

    return None;
  }

  auto accessPath =
    genericSig->getConformanceAccessPath(type, proto);

  // Fall through because we cannot yet evaluate an access path.
  Optional<ProtocolConformanceRef> conformance;
  for (const auto &step : accessPath) {
    // For the first step, grab the initial conformance.
    if (!conformance) {
      conformance = getInitialConformance(step.first, step.second);
      if (!conformance)
        return None;

      continue;
    }

    if (conformance->isInvalid())
      return conformance;

    // If we've hit an abstract conformance, everything from here on out is
    // abstract.
    // FIXME: This may not always be true, but it holds for now.
    if (conformance->isAbstract()) {
      // FIXME: Rip this out once we can get a concrete conformance from
      // an archetype.
      auto *M = proto->getParentModule();
      auto substType = type.subst(*this);
      if (substType &&
          (!substType->is<ArchetypeType>() ||
           substType->castTo<ArchetypeType>()->getSuperclass()) &&
          !substType->isTypeParameter() &&
          !substType->isExistentialType()) {
        return M->lookupConformance(substType, proto);
      }

      return ProtocolConformanceRef(proto);
    }

    // For the second step, we're looking into the requirement signature for
    // this protocol.
    auto concrete = conformance->getConcrete();
    auto normal = concrete->getRootNormalConformance();

    // If we haven't set the signature conformances yet, force the issue now.
    if (normal->getSignatureConformances().empty()) {
      // If we're in the process of checking the type witnesses, fail
      // gracefully.
      // FIXME: Seems like we should be able to get at the intermediate state
      // to use that.
      if (normal->getState() == ProtocolConformanceState::CheckingTypeWitnesses)
        return None;

      auto lazyResolver = type->getASTContext().getLazyResolver();
      if (lazyResolver == nullptr)
        return None;

      lazyResolver->resolveTypeWitness(normal, nullptr);

      // Error case: the conformance is broken, so we cannot handle this
      // substitution.
      if (normal->getSignatureConformances().empty())
        return None;
    }

    // Get the associated conformance.
    conformance = concrete->getAssociatedConformance(step.first, step.second);
  }

  return conformance;
}
