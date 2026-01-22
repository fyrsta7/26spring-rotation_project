bool Substitution::operator==(const Substitution &Other) const {
  return Archetype->getCanonicalType() == Other.Archetype->getCanonicalType() &&
    Replacement->getCanonicalType() == Other.Replacement->getCanonicalType() &&
    Conformance.equals(Other.Conformance);
}
