MaybeHandle<Object> KeyedStoreIC::Store(Handle<Object> object,
                                        Handle<Object> key,
                                        Handle<Object> value) {
  // TODO(verwaest): Let SetProperty do the migration, since storing a property
  // might deprecate the current map again, if value does not fit.
  if (MigrateDeprecated(object)) {
    Handle<Object> result;
    ASSIGN_RETURN_ON_EXCEPTION(
        isolate(), result, Runtime::SetObjectProperty(isolate(), object, key,
                                                      value, language_mode()),
        Object);
    return result;
  }

  // Check for non-string values that can be converted into an
  // internalized string directly or is representable as a smi.
  key = TryConvertKey(key, isolate());

  Handle<Object> store_handle;

  uint32_t index;
  if ((key->IsInternalizedString() &&
       !String::cast(*key)->AsArrayIndex(&index)) ||
      key->IsSymbol()) {
    ASSIGN_RETURN_ON_EXCEPTION(
        isolate(), store_handle,
        StoreIC::Store(object, Handle<Name>::cast(key), value,
                       JSReceiver::MAY_BE_STORE_FROM_KEYED),
        Object);
    if (!is_vector_set()) {
      ConfigureVectorState(MEGAMORPHIC, key);
      TRACE_GENERIC_IC("unhandled internalized string key");
      TRACE_IC("StoreIC", key);
    }
    return store_handle;
  }

  bool use_ic = FLAG_use_ic && !object->IsStringWrapper() &&
                !object->IsAccessCheckNeeded() && !object->IsJSGlobalProxy();
  if (use_ic && !object->IsSmi()) {
    // Don't use ICs for maps of the objects in Array's prototype chain. We
    // expect to be able to trap element sets to objects with those maps in
    // the runtime to enable optimization of element hole access.
    Handle<HeapObject> heap_object = Handle<HeapObject>::cast(object);
    if (heap_object->map()->IsMapInArrayPrototypeChain()) {
      TRACE_GENERIC_IC("map in array prototype");
      use_ic = false;
    }
  }

  Handle<Map> old_receiver_map;
  bool is_arguments = false;
  bool key_is_valid_index = false;
  KeyedAccessStoreMode store_mode = STANDARD_STORE;
  if (use_ic && object->IsJSObject()) {
    Handle<JSObject> receiver = Handle<JSObject>::cast(object);
    old_receiver_map = handle(receiver->map(), isolate());
    is_arguments = receiver->IsJSArgumentsObject();
    if (!is_arguments) {
      key_is_valid_index = key->IsSmi() && Smi::cast(*key)->value() >= 0;
      if (key_is_valid_index) {
        uint32_t index = static_cast<uint32_t>(Smi::cast(*key)->value());
        store_mode = GetStoreMode(receiver, index, value);
      }
    }
  }

  DCHECK(store_handle.is_null());
  ASSIGN_RETURN_ON_EXCEPTION(isolate(), store_handle,
                             Runtime::SetObjectProperty(isolate(), object, key,
                                                        value, language_mode()),
                             Object);

  if (use_ic) {
    if (!old_receiver_map.is_null()) {
      if (is_arguments) {
        TRACE_GENERIC_IC("arguments receiver");
      } else if (key_is_valid_index) {
        // We should go generic if receiver isn't a dictionary, but our
        // prototype chain does have dictionary elements. This ensures that
        // other non-dictionary receivers in the polymorphic case benefit
        // from fast path keyed stores.
        if (!old_receiver_map->DictionaryElementsInPrototypeChainOnly()) {
          UpdateStoreElement(old_receiver_map, store_mode);
        } else {
          TRACE_GENERIC_IC("dictionary or proxy prototype");
        }
      } else {
        TRACE_GENERIC_IC("non-smi-like key");
      }
    } else {
      TRACE_GENERIC_IC("non-JSObject receiver");
    }
  }

  if (!is_vector_set()) {
    ConfigureVectorState(MEGAMORPHIC, key);
  }
  TRACE_IC("StoreIC", key);

  return store_handle;
}
