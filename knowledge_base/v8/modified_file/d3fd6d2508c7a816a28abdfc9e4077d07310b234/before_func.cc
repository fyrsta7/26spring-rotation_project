Handle<Object> StackTraceFrame::GetFileName(Handle<StackTraceFrame> frame) {
  auto name = GetFrameInfo(frame)->script_name();
  return handle(name, frame->GetIsolate());
}
