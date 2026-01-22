Handle<Object> StackTraceFrame::GetScriptNameOrSourceUrl(
    Handle<StackTraceFrame> frame) {
  auto name = GetFrameInfo(frame)->script_name_or_source_url();
  return handle(name, frame->GetIsolate());
}
