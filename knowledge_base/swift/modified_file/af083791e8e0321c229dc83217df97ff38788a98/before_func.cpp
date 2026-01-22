Type ProtocolConformanceRef::getAssociatedType(Type conformingType,
                                               Type assocType,
                                               LazyResolver *resolver) const {
  assert(!isConcrete() || getConcrete()->getType()->isEqual(conformingType));

  auto type = assocType->getCanonicalType();
  auto proto = getRequirement();

#if false
  // Fast path for generic parameters.
  if (isa<GenericTypeParamType>(type)) {
    assert(type->isEqual(proto->getSelfInterfaceType()) &&
           "type parameter in protocol was not Self");
    return getType();
  }

  // Fast path for dependent member types on 'Self' of our associated types.
  auto memberType = cast<DependentMemberType>(type);
  if (memberType.getBase()->isEqual(proto->getProtocolSelfType()) &&
      memberType->getAssocType()->getProtocol() == proto)
    return getTypeWitness(memberType->getAssocType(), nullptr);
#endif

  // General case: consult the substitution map.
  auto substMap =
    SubstitutionMap::getProtocolSubstitutions(proto, conformingType, *this);
  return type.subst(substMap);
}
