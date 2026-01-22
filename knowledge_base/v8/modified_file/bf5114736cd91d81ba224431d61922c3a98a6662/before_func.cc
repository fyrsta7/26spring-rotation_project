void Scheduler::ScheduleLate() {
  if (FLAG_trace_turbo_scheduler) {
    PrintF("------------------- SCHEDULE LATE -----------------\n");
  }

  // Schedule: Places nodes in dominator block of all their uses.
  ScheduleLateNodeVisitor schedule_late_visitor(this);

  for (NodeVectorIter i = schedule_root_nodes_.begin();
       i != schedule_root_nodes_.end(); ++i) {
    // TODO(mstarzinger): Make the scheduler eat less memory.
    Zone zone(zone_->isolate());
    GenericGraphVisit::Visit<ScheduleLateNodeVisitor,
                             NodeInputIterationTraits<Node> >(
        graph_, &zone, *i, &schedule_late_visitor);
  }

  // Add collected nodes for basic blocks to their blocks in the right order.
  int block_num = 0;
  for (NodeVectorVectorIter i = scheduled_nodes_.begin();
       i != scheduled_nodes_.end(); ++i) {
    for (NodeVectorRIter j = i->rbegin(); j != i->rend(); ++j) {
      schedule_->AddNode(schedule_->all_blocks_.at(block_num), *j);
    }
    block_num++;
  }
}
