  return Status::OK();
}

Status DirectSession::Extend(const GraphDef& graph) {
  return Extend(GraphDef(graph));
}

Status DirectSession::Extend(GraphDef&& graph) {
  TF_RETURN_IF_ERROR(CheckNotClosed());
  mutex_lock l(graph_state_lock_);
  return ExtendLocked(std::move(graph));
}

Status DirectSession::ExtendLocked(GraphDef graph) {
  if (finalized_) {
    return errors::FailedPrecondition("Session has been finalized.");
  }
  if (!(flib_def_ && execution_state_)) {
    // If this is the first call, we can initialize the execution state
    // with `graph` and do not need to call `Extend()`.
    GraphExecutionStateOptions options;
    options.device_set = &device_set_;
    options.session_options = &options_;
    options.session_handle = session_handle_;
    TF_RETURN_IF_ERROR(GraphExecutionState::MakeForBaseGraph(
        std::move(graph), options, &execution_state_));
    // NOTE(mrry): The function library created here will be used for
    // all subsequent extensions of the graph. Also, note how using the copy
    // constructor of FunctionLibraryDefinition avoids duplicating the memory
    // that is occupied by its shared_ptr members.
