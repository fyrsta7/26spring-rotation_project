Handle<Object> StackTraceFrame::GetFileName(Handle<StackTraceFrame> frame) {
  Isolate* isolate = frame->GetIsolate();

  // Use FrameInfo if it's already there, but avoid initializing it for just
  // the file name, as it is much more expensive than just getting this
  // directly. See GetScriptNameOrSourceUrl() for more detail.
  if (!frame->frame_info().IsUndefined()) {
    auto name = GetFrameInfo(frame)->script_name();
    return handle(name, isolate);
  }
  FrameArrayIterator it(isolate,
                        handle(FrameArray::cast(frame->frame_array()), isolate),
                        frame->frame_index());
  DCHECK(it.HasFrame());
  return it.Frame()->GetFileName();
}
