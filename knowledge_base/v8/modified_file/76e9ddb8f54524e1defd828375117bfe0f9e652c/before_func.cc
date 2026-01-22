int StackTraceFrame::GetScriptId(Handle<StackTraceFrame> frame) {
  int id = GetFrameInfo(frame)->script_id();
  return id != StackFrameBase::kNone ? id : Message::kNoScriptIdInfo;
}
