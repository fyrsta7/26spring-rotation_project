Handle<Object> StackTraceFrame::GetScriptNameOrSourceUrl(
    Handle<StackTraceFrame> frame) {
  Isolate* isolate = frame->GetIsolate();
  // TODO(caseq, szuend): the logic below is a workaround for crbug.com/1057211.
  // We should probably have a dedicated API for the scenario described in the
  // bug above and make getters of this class behave consistently.
  // See https://bit.ly/2wkbuIy for further discussion.
  // Use FrameInfo if it's already there, but avoid initializing it for just
  // the script name, as it is much more expensive than just getting this
  // directly.
  if (!frame->frame_info().IsUndefined()) {
    auto name = GetFrameInfo(frame)->script_name_or_source_url();
    return handle(name, isolate);
  }
  FrameArrayIterator it(isolate,
                        handle(FrameArray::cast(frame->frame_array()), isolate),
                        frame->frame_index());
  DCHECK(it.HasFrame());
  return it.Frame()->GetScriptNameOrSourceUrl();
}
