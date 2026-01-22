bool CompilerDispatcher::IsEnqueued(Handle<SharedFunctionInfo> function) const {
  if (jobs_.empty()) return false;
  return GetJobFor(function) != jobs_.end();
}
