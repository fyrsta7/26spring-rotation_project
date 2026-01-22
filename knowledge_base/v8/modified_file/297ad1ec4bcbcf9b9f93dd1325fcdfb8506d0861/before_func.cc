Maybe<bool> ValueSerializer::WriteJSObject(Handle<JSObject> object) {
  DCHECK(!object->map().IsCustomElementsReceiverMap());
  const bool can_serialize_fast =
      object->HasFastProperties() && object->elements().length() == 0;
  if (!can_serialize_fast) return WriteJSObjectSlow(object);

  Handle<Map> map(object->map(), isolate_);
  WriteTag(SerializationTag::kBeginJSObject);

  // Write out fast properties as long as they are only data properties and the
  // map doesn't change.
  uint32_t properties_written = 0;
  bool map_changed = false;
  for (InternalIndex i : map->IterateOwnDescriptors()) {
    Handle<Name> key(map->instance_descriptors(kRelaxedLoad).GetKey(i),
                     isolate_);
    if (!key->IsString()) continue;
    PropertyDetails details =
        map->instance_descriptors(kRelaxedLoad).GetDetails(i);
    if (details.IsDontEnum()) continue;

    Handle<Object> value;
    if (V8_LIKELY(!map_changed)) map_changed = *map == object->map();
    if (V8_LIKELY(!map_changed && details.location() == kField)) {
      DCHECK_EQ(kData, details.kind());
      FieldIndex field_index = FieldIndex::ForDescriptor(*map, i);
      value = JSObject::FastPropertyAt(object, details.representation(),
                                       field_index);
    } else {
      // This logic should essentially match WriteJSObjectPropertiesSlow.
      // If the property is no longer found, do not serialize it.
      // This could happen if a getter deleted the property.
      LookupIterator it(isolate_, object, key, LookupIterator::OWN);
      if (!it.IsFound()) continue;
      if (!Object::GetProperty(&it).ToHandle(&value)) return Nothing<bool>();
    }

    if (!WriteObject(key).FromMaybe(false) ||
        !WriteObject(value).FromMaybe(false)) {
      return Nothing<bool>();
    }
    properties_written++;
  }

  WriteTag(SerializationTag::kEndJSObject);
  WriteVarint<uint32_t>(properties_written);
  return ThrowIfOutOfMemory();
}
