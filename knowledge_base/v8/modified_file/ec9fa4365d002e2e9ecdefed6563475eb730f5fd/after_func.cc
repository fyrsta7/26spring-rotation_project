void Shell::NodeTypeCallback(const v8::FunctionCallbackInfo<v8::Value>& args) {
  v8::Isolate* isolate = args.GetIsolate();

  // HasInstance does a slow prototype chain lookup, and this function is used
  // for micro benchmarks too.
#ifdef DEBUG
  PerIsolateData* data = PerIsolateData::Get(isolate);
  if (!data->GetDomNodeCtor()->HasInstance(args.This())) {
    isolate->ThrowError("Calling .nodeType on wrong instance type.");
  }
#endif

  args.GetReturnValue().Set(v8::Number::New(isolate, 1));
}
