std::pair<MaybeObject, MaybeObject> NexusConfig::GetFeedbackPair(
    FeedbackVector vector, FeedbackSlot slot) const {
  if (mode() == BackgroundThread) {
    isolate()->feedback_vector_access()->LockShared();
  }
  MaybeObject feedback = vector.Get(slot);
  MaybeObject feedback_extra = vector.Get(slot.WithOffset(1));
  auto return_value = std::make_pair(feedback, feedback_extra);
  if (mode() == BackgroundThread) {
    isolate()->feedback_vector_access()->UnlockShared();
  }
  return return_value;
}
