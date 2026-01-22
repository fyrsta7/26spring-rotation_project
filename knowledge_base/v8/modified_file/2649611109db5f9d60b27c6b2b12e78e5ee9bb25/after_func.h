MaybeHandle<JSReceiver> Object::ToObject(Isolate* isolate,
                                         Handle<Object> object) {
  if (object->IsJSReceiver()) return Handle<JSReceiver>::cast(object);
  return ToObject(isolate, object, isolate->native_context());
}
