bool IC::ConfigureVectorState(IC::State new_state, Handle<Object> key) {
  DCHECK_EQ(MEGAMORPHIC, new_state);
  DCHECK_IMPLIES(!is_keyed(), key->IsName());
  // Even though we don't change the feedback data, we still want to reset the
  // profiler ticks. Real-world observations suggest that optimizing these
  // functions doesn't improve performance.
  bool changed = nexus()->ConfigureMegamorphic(
      key->IsName() ? IcCheckType::kProperty : IcCheckType::kElement);
  OnFeedbackChanged("Megamorphic");
  return changed;
}
