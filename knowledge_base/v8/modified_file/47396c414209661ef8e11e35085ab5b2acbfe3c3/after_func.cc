void JSInliningHeuristic::Finalize() {
  if (candidates_.empty()) return;  // Nothing to do without candidates.
  if (FLAG_trace_turbo_inlining) PrintCandidates();

  // We inline at most one candidate in every iteration of the fixpoint.
  // This is to ensure that we don't consume the full inlining budget
  // on things that aren't called very often.
  // TODO(bmeurer): Use std::priority_queue instead of std::set here.
  while (!candidates_.empty()) {
    if (cumulative_count_ > FLAG_max_inlined_nodes_cumulative) return;
    auto i = candidates_.begin();
    Candidate candidate = *i;
    candidates_.erase(i);
    Reduction r = inliner_.ReduceJSCall(candidate.node, candidate.function);
    if (r.Changed()) {
      cumulative_count_ += candidate.function->shared()->ast_node_count();
      return;
    }
  }
}
