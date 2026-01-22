int StackTraceFrame::GetScriptId(Handle<StackTraceFrame> frame) {
  Isolate* isolate = frame->GetIsolate();

  // Use FrameInfo if it's already there, but avoid initializing it for just
  // the script id, as it is much more expensive than just getting this
  // directly. See GetScriptNameOrSourceUrl() for more detail.
  int id;
  if (!frame->frame_info().IsUndefined()) {
    id = GetFrameInfo(frame)->script_id();
  } else {
    FrameArrayIterator it(
        isolate, handle(FrameArray::cast(frame->frame_array()), isolate),
        frame->frame_index());
    DCHECK(it.HasFrame());
    id = it.Frame()->GetScriptId();
  }
  return id != StackFrameBase::kNone ? id : Message::kNoScriptIdInfo;
}
