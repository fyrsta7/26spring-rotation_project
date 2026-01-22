MaybeHandle<String> WasmStackFrame::ToString() {
  IncrementalStringBuilder builder(isolate_);

  Handle<Object> name = GetFunctionName();
  if (name->IsNull(isolate_)) {
    builder.AppendCString("<WASM UNNAMED>");
  } else {
    DCHECK(name->IsString());
    builder.AppendString(Handle<String>::cast(name));
  }

  builder.AppendCString(" (<WASM>[");

  char buffer[16];
  SNPrintF(ArrayVector(buffer), "%u", wasm_func_index_);
  builder.AppendCString(buffer);

  builder.AppendCString("]+");

  SNPrintF(ArrayVector(buffer), "%d", GetPosition());
  builder.AppendCString(buffer);
  builder.AppendCString(")");

  return builder.Finish();
}
