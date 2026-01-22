TF_BUILTIN(ObjectKeys, ObjectBuiltinsAssembler) {
  Node* object = Parameter(Descriptor::kObject);
  Node* context = Parameter(Descriptor::kContext);

  VARIABLE(var_length, MachineRepresentation::kTagged);
  VARIABLE(var_elements, MachineRepresentation::kTagged);
  Label if_empty(this, Label::kDeferred), if_fast(this),
      if_slow(this, Label::kDeferred), if_join(this);

  // Check if the {object} has a usable enum cache.
  GotoIf(TaggedIsSmi(object), &if_slow);
  Node* object_map = LoadMap(object);
  Node* object_bit_field3 = LoadMapBitField3(object_map);
  Node* object_enum_length =
      DecodeWordFromWord32<Map::EnumLengthBits>(object_bit_field3);
  GotoIf(
      WordEqual(object_enum_length, IntPtrConstant(kInvalidEnumCacheSentinel)),
      &if_slow);

  // Ensure that the {object} doesn't have any elements.
  CSA_ASSERT(this, IsJSObjectMap(object_map));
  Node* object_elements = LoadObjectField(object, JSObject::kElementsOffset);
  GotoIfNot(IsEmptyFixedArray(object_elements), &if_slow);
  Branch(WordEqual(object_enum_length, IntPtrConstant(0)), &if_empty, &if_fast);

  BIND(&if_fast);
  {
    // The {object} has a usable enum cache, use that.
    Node* object_descriptors = LoadMapDescriptors(object_map);
    Node* object_enum_cache =
        LoadObjectField(object_descriptors, DescriptorArray::kEnumCacheOffset);
    Node* object_enum_keys =
        LoadObjectField(object_enum_cache, EnumCache::kKeysOffset);

    // Allocate a JSArray and copy the elements from the {object_enum_keys}.
    Node* array = nullptr;
    Node* elements = nullptr;
    Node* native_context = LoadNativeContext(context);
    Node* array_map = LoadJSArrayElementsMap(PACKED_ELEMENTS, native_context);
    Node* array_length = SmiTag(object_enum_length);
    std::tie(array, elements) = AllocateUninitializedJSArrayWithElements(
        PACKED_ELEMENTS, array_map, array_length, nullptr, object_enum_length,
        INTPTR_PARAMETERS);
    CopyFixedArrayElements(PACKED_ELEMENTS, object_enum_keys, elements,
                           object_enum_length, SKIP_WRITE_BARRIER);
    Return(array);
  }

  BIND(&if_empty);
  {
    // The {object} doesn't have any enumerable keys.
    var_length.Bind(SmiConstant(0));
    var_elements.Bind(EmptyFixedArrayConstant());
    Goto(&if_join);
  }

  BIND(&if_slow);
  {
    // Let the runtime compute the elements.
    Node* elements = CallRuntime(Runtime::kObjectKeys, context, object);
    var_length.Bind(LoadObjectField(elements, FixedArray::kLengthOffset));
    var_elements.Bind(elements);
    Goto(&if_join);
  }

  BIND(&if_join);
  {
    // Wrap the elements into a proper JSArray and return that.
    Node* native_context = LoadNativeContext(context);
    Node* array_map = LoadJSArrayElementsMap(PACKED_ELEMENTS, native_context);
    Node* array = AllocateUninitializedJSArrayWithoutElements(
        PACKED_ELEMENTS, array_map, var_length.value(), nullptr);
    StoreObjectFieldNoWriteBarrier(array, JSArray::kElementsOffset,
                                   var_elements.value());
    Return(array);
  }
}
