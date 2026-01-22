std::pair<MaybeObject, MaybeObject> NexusConfig::GetFeedbackPair(
    FeedbackVector vector, FeedbackSlot slot) const {
  base::SharedMutexGuard<base::kShared> shared_mutex_guard(
      isolate()->feedback_vector_access());
  MaybeObject feedback = vector.Get(slot);
  MaybeObject feedback_extra = vector.Get(slot.WithOffset(1));
  return std::make_pair(feedback, feedback_extra);
}
