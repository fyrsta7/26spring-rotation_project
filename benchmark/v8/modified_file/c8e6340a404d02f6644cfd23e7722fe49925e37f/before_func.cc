bool CompilerDispatcher::IsEnqueued(Handle<SharedFunctionInfo> function) const {
  return GetJobFor(function) != jobs_.end();
}