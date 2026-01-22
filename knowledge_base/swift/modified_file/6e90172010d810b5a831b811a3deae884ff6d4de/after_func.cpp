ConcreteDeclRef
SpecializedProtocolConformance::getWitnessDeclRef(
                                              ValueDecl *requirement,
                                              LazyResolver *resolver) const {
  auto baseWitness = GenericConformance->getWitnessDeclRef(requirement, resolver);
  if (!baseWitness || !baseWitness.isSpecialized())
    return baseWitness;

  auto specializationMap = getSubstitutionMap();

  auto witnessDecl = baseWitness.getDecl();
  auto witnessMap = baseWitness.getSubstitutions();

  auto combinedMap = witnessMap.subst(specializationMap);
  return ConcreteDeclRef(witnessDecl, combinedMap);
}
