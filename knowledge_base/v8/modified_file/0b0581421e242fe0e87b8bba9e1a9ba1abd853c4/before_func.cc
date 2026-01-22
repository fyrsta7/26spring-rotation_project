void JSInliningHeuristic::Finalize() {
  if (candidates_.empty()) return;  // Nothing to do without candidates.
  if (FLAG_trace_turbo_inlining) PrintCandidates();

  while (!candidates_.empty()) {
    if (cumulative_count_ > FLAG_max_inlined_nodes_cumulative) break;
    auto i = candidates_.begin();
    Candidate const& candidate = *i;
    inliner_.ReduceJSCall(candidate.node, candidate.function);
    cumulative_count_ += candidate.function->shared()->ast_node_count();
    candidates_.erase(i);
  }
}
