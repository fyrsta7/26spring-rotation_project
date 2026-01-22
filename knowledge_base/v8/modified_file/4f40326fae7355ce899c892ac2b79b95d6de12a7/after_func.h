int TransitionArray::Search(String* name) {
  if (IsSimpleTransition()) {
    String* key = GetKey(kSimpleTransitionIndex);
    if (key->Equals(name)) return kSimpleTransitionIndex;
    return kNotFound;
  }
  return internal::Search<ALL_ENTRIES>(this, name);
}
