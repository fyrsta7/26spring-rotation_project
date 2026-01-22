template <typename Char>
Handle<Object> JsonParser<Char>::BuildJsonObject(
    const JsonContinuation& cont,
    const SmallVector<JsonProperty>& property_stack, Handle<Map> feedback) {
  size_t start = cont.index;
  DCHECK_LE(start, property_stack.size());
  int length = static_cast<int>(property_stack.size() - start);
  int named_length = length - cont.elements;
  DCHECK_LE(0, named_length);

  Handle<FixedArrayBase> elements;
  ElementsKind elements_kind = HOLEY_ELEMENTS;

  // First store the elements.
  if (cont.elements > 0) {
    // Store as dictionary elements if that would use less memory.
    if (ShouldConvertToSlowElements(cont.elements, cont.max_index + 1)) {
      Handle<NumberDictionary> elms =
          NumberDictionary::New(isolate_, cont.elements);
      for (int i = 0; i < length; i++) {
        const JsonProperty& property = property_stack[start + i];
        if (!property.string.is_index()) continue;
        uint32_t index = property.string.index();
        Handle<Object> value = property.value;
        NumberDictionary::UncheckedSet(isolate_, elms, index, value);
      }
      elms->SetInitialNumberOfElements(length);
      elms->UpdateMaxNumberKey(cont.max_index, Handle<JSObject>::null());
      elements_kind = DICTIONARY_ELEMENTS;
      elements = elms;
    } else {
      Handle<FixedArray> elms =
          factory()->NewFixedArrayWithHoles(cont.max_index + 1);
      DisallowGarbageCollection no_gc;
      Tagged<FixedArray> raw_elements = *elms;
      WriteBarrierMode mode = raw_elements->GetWriteBarrierMode(no_gc);

      for (int i = 0; i < length; i++) {
        const JsonProperty& property = property_stack[start + i];
        if (!property.string.is_index()) continue;
        uint32_t index = property.string.index();
        Handle<Object> value = property.value;
        raw_elements->set(static_cast<int>(index), *value, mode);
      }
      elements = elms;
    }
  } else {
    elements = factory()->empty_fixed_array();
  }

  JSDataObjectBuilder js_data_object_builder(
      isolate_, elements_kind, named_length, feedback,
      JSDataObjectBuilder::kHeapNumbersGuaranteedUniquelyOwned);

  Handle<String> failed_property_add_key;
  int i;
  for (i = 0; i < length; i++) {
    const JsonProperty& property = property_stack[start + i];
    if (property.string.is_index()) continue;

    Handle<String> property_key;
    if (!js_data_object_builder.TryAddFastPropertyForValue(
            [&](Handle<String> expected_key) {
              return property_key = MakeString(property.string, expected_key);
            },
            [&]() { return property.value; })) {
      failed_property_add_key = property_key;
      break;
    }
  }

  // Iterator for re-visiting the values that were visited by the above fast
  // path property initialisation.
  NamedPropertyValueIterator named_property_values(
      property_stack.begin() + start, property_stack.end());
  Handle<JSObject> object = js_data_object_builder.CreateAndInitialiseObject(
      named_property_values, elements);

  // Slow path: define remaining named properties.
  for (; i < length; i++) {
    HandleScope scope(isolate_);
    const JsonProperty& property = property_stack[start + i];
    if (property.string.is_index()) continue;
    Handle<String> key;
    if (!failed_property_add_key.is_null()) {
      key = std::exchange(failed_property_add_key, {});
    } else {
      key = MakeString(property.string);
    }
#ifdef DEBUG
    uint32_t index;
    DCHECK(!key->AsArrayIndex(&index));
#endif
    Handle<Object> value = property.value;
    LookupIterator it(isolate_, object, key, object, LookupIterator::OWN);
    JSObject::DefineOwnPropertyIgnoreAttributes(&it, value, NONE).Check();
  }

  return object;
}
