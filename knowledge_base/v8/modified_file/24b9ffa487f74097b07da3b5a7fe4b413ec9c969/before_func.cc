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

  Handle<Smi> ix(Smi::FromInt(wasm_func_index_), isolate_);
  builder.AppendString(isolate_->factory()->NumberToString(ix));

  builder.AppendCString("]+");

  Handle<Object> pos(Smi::FromInt(GetPosition()), isolate_);
  builder.AppendString(isolate_->factory()->NumberToString(pos));
  builder.AppendCString(")");

  return builder.Finish();
}
