Maybe<bool> JSReceiver::SetOrCopyDataProperties(
    Isolate* isolate, Handle<JSReceiver> target, Handle<Object> source,
    const ScopedVector<Handle<Object>>* excluded_properties, bool use_set) {
  Maybe<bool> fast_assign =
      FastAssign(target, source, excluded_properties, use_set);
  if (fast_assign.IsNothing()) return Nothing<bool>();
  if (fast_assign.FromJust()) return Just(true);

  Handle<JSReceiver> from = Object::ToObject(isolate, source).ToHandleChecked();

  // 3b. Let keys be ? from.[[OwnPropertyKeys]]().
  Handle<FixedArray> keys;
  ASSIGN_RETURN_ON_EXCEPTION_VALUE(
      isolate, keys,
      KeyAccumulator::GetKeys(from, KeyCollectionMode::kOwnOnly, ALL_PROPERTIES,
                              GetKeysConversion::kKeepNumbers),
      Nothing<bool>());

  if (!from->HasFastProperties() && target->HasFastProperties()) {
    // Convert to slow properties if we're guaranteed to overflow the number of
    // descriptors.
    int source_length =
        from->property_dictionary().NumberOfEnumerableProperties();
    if (source_length > kMaxNumberOfDescriptors) {
      JSObject::NormalizeProperties(isolate, Handle<JSObject>::cast(target),
                                    CLEAR_INOBJECT_PROPERTIES, source_length,
                                    "Copying data properties");
    }
  }

  // 4. Repeat for each element nextKey of keys in List order,
  for (int i = 0; i < keys->length(); ++i) {
    Handle<Object> next_key(keys->get(i), isolate);
    // 4a i. Let desc be ? from.[[GetOwnProperty]](nextKey).
    PropertyDescriptor desc;
    Maybe<bool> found =
        JSReceiver::GetOwnPropertyDescriptor(isolate, from, next_key, &desc);
    if (found.IsNothing()) return Nothing<bool>();
    // 4a ii. If desc is not undefined and desc.[[Enumerable]] is true, then
    if (found.FromJust() && desc.enumerable()) {
      // 4a ii 1. Let propValue be ? Get(from, nextKey).
      Handle<Object> prop_value;
      ASSIGN_RETURN_ON_EXCEPTION_VALUE(
          isolate, prop_value,
          Runtime::GetObjectProperty(isolate, from, next_key), Nothing<bool>());

      if (use_set) {
        // 4c ii 2. Let status be ? Set(to, nextKey, propValue, true).
        Handle<Object> status;
        ASSIGN_RETURN_ON_EXCEPTION_VALUE(
            isolate, status,
            Runtime::SetObjectProperty(isolate, target, next_key, prop_value,
                                       StoreOrigin::kMaybeKeyed,
                                       Just(ShouldThrow::kThrowOnError)),
            Nothing<bool>());
      } else {
        if (excluded_properties != nullptr &&
            HasExcludedProperty(excluded_properties, next_key)) {
          continue;
        }

        // 4a ii 2. Perform ! CreateDataProperty(target, nextKey, propValue).
        LookupIterator::Key key(isolate, next_key);
        LookupIterator it(isolate, target, key, LookupIterator::OWN);
        CHECK(JSObject::CreateDataProperty(&it, prop_value, Just(kThrowOnError))
                  .FromJust());
      }
    }
  }

  return Just(true);
}
