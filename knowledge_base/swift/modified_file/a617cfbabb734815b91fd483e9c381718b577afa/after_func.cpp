bool Substitution::operator==(const Substitution &Other) const {
  // The archetypes may be missing, but we can compare them directly
  // because archetypes are always canonical.
  return Archetype == Other.Archetype &&
    Replacement->getCanonicalType() == Other.Replacement->getCanonicalType() &&
    Conformance.equals(Other.Conformance);
}
