Maybe<bool> KeyAccumulator::CollectOwnPropertyNames(Handle<JSReceiver> receiver,
                                                    Handle<JSObject> object) {
  if (filter_ == ENUMERABLE_STRINGS) {
    Handle<FixedArray> enum_keys;
    if (object->HasFastProperties()) {
      enum_keys = KeyAccumulator::GetOwnEnumPropertyKeys(isolate_, object);
      // If the number of properties equals the length of enumerable properties
      // we do not have to filter out non-enumerable ones
      Map map = object->map();
      int nof_descriptors = map.NumberOfOwnDescriptors();
      if (enum_keys->length() != nof_descriptors) {
        Handle<DescriptorArray> descs =
            Handle<DescriptorArray>(map.instance_descriptors(), isolate_);
        for (int i = 0; i < nof_descriptors; i++) {
          PropertyDetails details = descs->GetDetails(i);
          if (!details.IsDontEnum()) continue;
          Object key = descs->GetKey(i);
          this->AddShadowingKey(key);
        }
      }
    } else if (object->IsJSGlobalObject()) {
      enum_keys = GetOwnEnumPropertyDictionaryKeys(
          isolate_, mode_, this, object,
          JSGlobalObject::cast(*object).global_dictionary());
    } else {
      enum_keys = GetOwnEnumPropertyDictionaryKeys(
          isolate_, mode_, this, object, object->property_dictionary());
    }
    if (object->IsJSModuleNamespace()) {
      // Simulate [[GetOwnProperty]] for establishing enumerability, which
      // throws for uninitialized exports.
      for (int i = 0, n = enum_keys->length(); i < n; ++i) {
        Handle<String> key(String::cast(enum_keys->get(i)), isolate_);
        if (Handle<JSModuleNamespace>::cast(object)
                ->GetExport(isolate(), key)
                .is_null()) {
          return Nothing<bool>();
        }
      }
    }
    RETURN_NOTHING_IF_NOT_SUCCESSFUL(AddKeys(enum_keys, DO_NOT_CONVERT));
  } else {
    if (object->HasFastProperties()) {
      int limit = object->map().NumberOfOwnDescriptors();
      Handle<DescriptorArray> descs(object->map().instance_descriptors(),
                                    isolate_);
      // First collect the strings,
      base::Optional<int> first_symbol =
          CollectOwnPropertyNamesInternal<true>(object, this, descs, 0, limit);
      // then the symbols.
      RETURN_NOTHING_IF_NOT_SUCCESSFUL(first_symbol);
      if (first_symbol.value() != -1) {
        RETURN_NOTHING_IF_NOT_SUCCESSFUL(CollectOwnPropertyNamesInternal<false>(
            object, this, descs, first_symbol.value(), limit));
      }
    } else if (object->IsJSGlobalObject()) {
      RETURN_NOTHING_IF_NOT_SUCCESSFUL(GlobalDictionary::CollectKeysTo(
          handle(JSGlobalObject::cast(*object).global_dictionary(), isolate_),
          this));
    } else {
      RETURN_NOTHING_IF_NOT_SUCCESSFUL(NameDictionary::CollectKeysTo(
          handle(object->property_dictionary(), isolate_), this));
    }
  }
  // Add the property keys from the interceptor.
  return CollectInterceptorKeys(receiver, object, this, kNamed);
}
