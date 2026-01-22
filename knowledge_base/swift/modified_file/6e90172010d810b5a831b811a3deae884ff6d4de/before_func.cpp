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

  // Fast path if the substitutions didn't change.
  if (combinedMap == baseWitness.getSubstitutions())
    return baseWitness;

  return ConcreteDeclRef(witnessDecl, combinedMap);
}
