void Shell::NodeTypeCallback(const v8::FunctionCallbackInfo<v8::Value>& args) {
  v8::Isolate* isolate = args.GetIsolate();
  // TODO(mslekova): Enable this once we have signature check in TF.
  PerIsolateData* data = PerIsolateData::Get(isolate);
  if (!data->GetDomNodeCtor()->HasInstance(args.This())) {
    isolate->ThrowError("Calling .nodeType on wrong instance type.");
  }

  args.GetReturnValue().Set(v8::Number::New(isolate, 1));
}
