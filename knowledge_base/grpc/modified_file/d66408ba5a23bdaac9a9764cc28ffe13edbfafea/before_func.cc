  Local<FunctionTemplate> tpl = FunctionTemplate::New(New);
  tpl->SetClassName(NanNew("Call"));
  tpl->InstanceTemplate()->SetInternalFieldCount(1);
  NanSetPrototypeTemplate(tpl, "startBatch",
                          FunctionTemplate::New(StartBatch)->GetFunction());
  NanSetPrototypeTemplate(tpl, "cancel",
                          FunctionTemplate::New(Cancel)->GetFunction());
  NanAssignPersistent(fun_tpl, tpl);
  NanAssignPersistent(constructor, tpl->GetFunction());
  constructor->Set(NanNew("WRITE_BUFFER_HINT"),
                   NanNew<Uint32, uint32_t>(GRPC_WRITE_BUFFER_HINT));
  constructor->Set(NanNew("WRITE_NO_COMPRESS"),
                   NanNew<Uint32, uint32_t>(GRPC_WRITE_NO_COMPRESS));
  exports->Set(String::NewSymbol("Call"), constructor);
}

bool Call::HasInstance(Handle<Value> val) {
  NanScope();
  return NanHasInstance(fun_tpl, val);
}

Handle<Value> Call::WrapStruct(grpc_call *call) {
  NanEscapableScope();
  if (call == NULL) {
    return NanEscapeScope(NanNull());
  }
  const int argc = 1;
  Handle<Value> argv[argc] = {External::New(reinterpret_cast<void *>(call))};
  return NanEscapeScope(constructor->NewInstance(argc, argv));
}

NAN_METHOD(Call::New) {
  NanScope();

  if (args.IsConstructCall()) {
    Call *call;
    if (args[0]->IsExternal()) {
      // This option is used for wrapping an existing call
      grpc_call *call_value =
          reinterpret_cast<grpc_call *>(External::Unwrap(args[0]));
      call = new Call(call_value);
    } else {
      if (!Channel::HasInstance(args[0])) {
        return NanThrowTypeError("Call's first argument must be a Channel");
