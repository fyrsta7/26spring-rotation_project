HValue* HGraphBuilder::BuildAllocateEmptyArrayBuffer(HValue* byte_length) {
  // We HForceRepresentation here to avoid allocations during an *-to-tagged
  // HChange that could cause GC while the array buffer object is not fully
  // initialized.
  HObjectAccess byte_length_access(HObjectAccess::ForJSArrayBufferByteLength());
  byte_length = AddUncasted<HForceRepresentation>(
      byte_length, byte_length_access.representation());
  HAllocate* result =
      BuildAllocate(Add<HConstant>(JSArrayBuffer::kSizeWithInternalFields),
                    HType::JSObject(), JS_ARRAY_BUFFER_TYPE, HAllocationMode());

  HValue* global_object = Add<HLoadNamedField>(
      context(), nullptr,
      HObjectAccess::ForContextSlot(Context::GLOBAL_OBJECT_INDEX));
  HValue* native_context = Add<HLoadNamedField>(
      global_object, nullptr, HObjectAccess::ForGlobalObjectNativeContext());
  Add<HStoreNamedField>(
      result, HObjectAccess::ForMap(),
      Add<HLoadNamedField>(
          native_context, nullptr,
          HObjectAccess::ForContextSlot(Context::ARRAY_BUFFER_MAP_INDEX)));

  HConstant* empty_fixed_array =
      Add<HConstant>(isolate()->factory()->empty_fixed_array());
  Add<HStoreNamedField>(
      result, HObjectAccess::ForJSArrayOffset(JSArray::kPropertiesOffset),
      empty_fixed_array);
  Add<HStoreNamedField>(
      result, HObjectAccess::ForJSArrayOffset(JSArray::kElementsOffset),
      empty_fixed_array);
  Add<HStoreNamedField>(
      result, HObjectAccess::ForJSArrayBufferBackingStore().WithRepresentation(
                  Representation::Smi()),
      graph()->GetConstant0());
  Add<HStoreNamedField>(result, byte_length_access, byte_length);
  Add<HStoreNamedField>(result, HObjectAccess::ForJSArrayBufferBitFieldSlot(),
                        graph()->GetConstant0());
  Add<HStoreNamedField>(
      result, HObjectAccess::ForJSArrayBufferBitField(),
      Add<HConstant>((1 << JSArrayBuffer::IsExternal::kShift) |
                     (1 << JSArrayBuffer::IsNeuterable::kShift)));

  for (int field = 0; field < v8::ArrayBuffer::kInternalFieldCount; ++field) {
    Add<HStoreNamedField>(
        result,
        HObjectAccess::ForObservableJSObjectOffset(
            JSArrayBuffer::kSize + field * kPointerSize, Representation::Smi()),
        graph()->GetConstant0());
  }

  return result;
}
