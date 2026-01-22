void InstructionScheduler::SchedulingQueueBase::AddNode(
    ScheduleGraphNode* node) {
  // We keep the ready list sorted by total latency so that we can quickly find
  // the next best candidate to schedule.
  auto it = nodes_.begin();
  while ((it != nodes_.end()) &&
         ((*it)->total_latency() >= node->total_latency())) {
    ++it;
  }
  nodes_.insert(it, node);
}
