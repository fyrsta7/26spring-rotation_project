MaybeHandle<JSReceiver> Object::ToObject(Isolate* isolate,
                                         Handle<Object> object) {
  return ToObject(
      isolate, object, handle(isolate->context()->native_context(), isolate));
}
