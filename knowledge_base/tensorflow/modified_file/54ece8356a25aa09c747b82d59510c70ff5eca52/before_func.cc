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
    // NOTE(mrry): The function library created here will be used for
    // all subsequent extensions of the graph.
    flib_def_.reset(
        new FunctionLibraryDefinition(OpRegistry::Global(), graph.library()));
    GraphExecutionStateOptions options;
    options.device_set = &device_set_;
    options.session_options = &options_;
    options.session_handle = session_handle_;
