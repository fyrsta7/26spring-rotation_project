bool CanInlineArrayIteratingBuiltin(Handle<Map> receiver_map) {
  Isolate* const isolate = receiver_map->GetIsolate();
  if (!receiver_map->prototype()->IsJSArray()) return false;
  Handle<JSArray> receiver_prototype(JSArray::cast(receiver_map->prototype()),
                                     isolate);
  return receiver_map->instance_type() == JS_ARRAY_TYPE &&
         IsFastElementsKind(receiver_map->elements_kind()) &&
         (!receiver_map->is_prototype_map() || receiver_map->is_stable()) &&
         isolate->IsNoElementsProtectorIntact() &&
         isolate->IsAnyInitialArrayPrototype(receiver_prototype);
}
