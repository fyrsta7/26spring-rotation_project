      pending_sink_nodes(executor->sink().size()),
      abort(false) {
  DCHECK(runner == nullptr || static_cast<bool>(*runner))
      << "`runner` must be nullptr or a valid TaskRunner";

  Node* node = nodes.data();
  for (const NodeDef& node_def : executor->nodes_defs()) {
    node->counter.store(node_def.in_edges.size(), std::memory_order_release);
    node->out_edges = &node_def.out_edges;
    ++node;
  }
}

tsl::AsyncValueRef<ThunkExecutor::ExecuteEvent> ThunkExecutor::Execute(
    const Thunk::ExecuteParams& params) {
  // Short-circuit execution of trivial thunk sequences.
  if (ABSL_PREDICT_FALSE(num_thunks_ == 0)) {
    return Thunk::OkExecuteEventSingleton();
  }
  if (ABSL_PREDICT_FALSE(num_thunks_ == 1)) {
    return thunk_sequence_[0]->Execute(params);
  }

  // If thunk sequence dependencies form a sequential execution graph, we skip
  // expensive async execution and simply run thunks one by one.
  if (is_sequential_) {
    return ExecuteSequential(params);
  }

  // Create async execution state on heap and kick-off execution.
  auto state = std::make_unique<ExecuteState>(this, params.task_runner);
  Execute(state.get(), params, ReadyQueue(source_.begin(), source_.end()),
          /*lock=*/params.session.Join());

  // If execution already completed (all kernels executed in the caller thread),
  // immediately return the result to avoid wasteful reference counting below.
  if (ABSL_PREDICT_TRUE(state->execute_event.IsAvailable())) {
    return std::move(state->execute_event);
  }
