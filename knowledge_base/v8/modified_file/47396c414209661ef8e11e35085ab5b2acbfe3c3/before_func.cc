void JSInliningHeuristic::Finalize() {
  if (candidates_.empty()) return;  // Nothing to do without candidates.
  if (FLAG_trace_turbo_inlining) PrintCandidates();

  // We inline at most one candidate in every iteration of the fixpoint.
  // This is to ensure that we don't consume the full inlining budget
  // on things that aren't called very often.
  if (cumulative_count_ > FLAG_max_inlined_nodes_cumulative) return;
  auto i = candidates_.begin();
  Candidate const& candidate = *i;
  inliner_.ReduceJSCall(candidate.node, candidate.function);
  cumulative_count_ += candidate.function->shared()->ast_node_count();
  candidates_.erase(i);
}
