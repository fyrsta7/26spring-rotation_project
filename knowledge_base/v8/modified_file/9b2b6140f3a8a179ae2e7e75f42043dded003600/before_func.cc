void WebSnapshotDeserializer::DeserializeMaps() {
  RCS_SCOPE(isolate_, RuntimeCallCounterId::kWebSnapshotDeserialize_Maps);
  if (!deserializer_->ReadUint32(&map_count_) || map_count_ > kMaxItemCount) {
    Throw("Malformed shape table");
    return;
  }
  STATIC_ASSERT(kMaxItemCount <= FixedArray::kMaxLength);
  maps_ = isolate_->factory()->NewFixedArray(map_count_);
  for (uint32_t i = 0; i < map_count_; ++i) {
    uint32_t map_type;
    if (!deserializer_->ReadUint32(&map_type)) {
      Throw("Malformed shape");
      return;
    }
    bool has_custom_property_attributes;
    switch (map_type) {
      case PropertyAttributesType::DEFAULT:
        has_custom_property_attributes = false;
        break;
      case PropertyAttributesType::CUSTOM:
        has_custom_property_attributes = true;
        break;
      default:
        Throw("Unsupported map type");
        return;
    }

    uint32_t prototype_id;
    if (!deserializer_->ReadUint32(&prototype_id) ||
        prototype_id > kMaxItemCount) {
      Throw("Malformed shape");
      return;
    }

    uint32_t property_count;
    if (!deserializer_->ReadUint32(&property_count)) {
      Throw("Malformed shape");
      return;
    }
    // TODO(v8:11525): Consider passing the upper bound as a param and
    // systematically enforcing it on the ValueSerializer side.
    if (property_count > kMaxNumberOfDescriptors) {
      Throw("Malformed shape: too many properties");
      return;
    }

    if (property_count == 0) {
      DisallowGarbageCollection no_gc;
      Map empty_map =
          isolate_->native_context()->object_function().initial_map();
      maps_->set(i, empty_map);
      return;
    }

    Handle<DescriptorArray> descriptors =
        isolate_->factory()->NewDescriptorArray(0, property_count);
    for (uint32_t p = 0; p < property_count; ++p) {
      PropertyAttributes attributes = PropertyAttributes::NONE;
      if (has_custom_property_attributes) {
        uint32_t flags;
        if (!deserializer_->ReadUint32(&flags)) {
          Throw("Malformed shape");
          return;
        }
        attributes = FlagsToAttributes(flags);
      }

      Handle<String> key = ReadString(true);

      // Use the "none" representation until we see the first object having this
      // map. At that point, modify the representation.
      Descriptor desc =
          Descriptor::DataField(isolate_, key, static_cast<int>(p), attributes,
                                Representation::None());
      descriptors->Append(&desc);
    }

    Handle<Map> map = isolate_->factory()->NewMap(
        JS_OBJECT_TYPE, JSObject::kHeaderSize * kTaggedSize, HOLEY_ELEMENTS, 0);
    map->InitializeDescriptors(isolate_, *descriptors);
    // TODO(v8:11525): Set 'constructor'.

    if (prototype_id == 0) {
      // Use Object.prototype as the prototype.
      map->set_prototype(isolate_->context().initial_object_prototype(),
                         UPDATE_WRITE_BARRIER);
    } else {
      // TODO(v8::11525): Implement stricter checks, e.g., disallow cycles.
      --prototype_id;
      if (prototype_id < current_object_count_) {
        map->set_prototype(HeapObject::cast(objects_->get(prototype_id)),
                           UPDATE_WRITE_BARRIER);
      } else {
        // The object hasn't been deserialized yet.
        AddDeferredReference(map, 0, OBJECT_ID, prototype_id);
      }
    }
    maps_->set(i, *map);
  }
}
