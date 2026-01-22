HInstruction* HOptimizedGraphBuilder::BuildMonomorphicElementAccess(
    HValue* object,
    HValue* key,
    HValue* val,
    HValue* dependency,
    Handle<Map> map,
    PropertyAccessType access_type,
    KeyedAccessStoreMode store_mode) {
  HCheckMaps* checked_object = Add<HCheckMaps>(object, map, dependency);
  if (dependency) {
    checked_object->ClearDependsOnFlag(kElementsKind);
  }

  if (access_type == STORE && map->prototype()->IsJSObject()) {
    // monomorphic stores need a prototype chain check because shape
    // changes could allow callbacks on elements in the chain that
    // aren't compatible with monomorphic keyed stores.
    Handle<JSObject> prototype(JSObject::cast(map->prototype()));
    Object* holder = map->prototype();
    while (holder->GetPrototype(isolate())->IsJSObject()) {
      holder = holder->GetPrototype(isolate());
    }
    ASSERT(holder->GetPrototype(isolate())->IsNull());

    BuildCheckPrototypeMaps(prototype,
                            Handle<JSObject>(JSObject::cast(holder)));
  }

  LoadKeyedHoleMode load_mode = BuildKeyedHoleMode(map);
  return BuildUncheckedMonomorphicElementAccess(
      checked_object, key, val,
      map->instance_type() == JS_ARRAY_TYPE,
      map->elements_kind(), access_type,
      load_mode, store_mode);
}
