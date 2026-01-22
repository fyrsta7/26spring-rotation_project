bool IC::ConfigureVectorState(IC::State new_state, Handle<Object> key) {
  DCHECK_EQ(MEGAMORPHIC, new_state);
  DCHECK_IMPLIES(!is_keyed(), key->IsName());
  bool changed = nexus()->ConfigureMegamorphic(
      key->IsName() ? IcCheckType::kProperty : IcCheckType::kElement);
  if (changed) {
    OnFeedbackChanged("Megamorphic");
  }
  return changed;
}
