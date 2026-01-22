Node* CodeStubAssembler::CloneFastJSArray(Node* context, Node* array,
                                          ParameterMode mode,
                                          Node* allocation_site) {
  Node* original_array_map = LoadMap(array);
  Node* elements_kind = LoadMapElementsKind(original_array_map);

  Node* length = LoadJSArrayLength(array);
  Node* new_elements = ExtractFixedArray(
      LoadElements(array), IntPtrOrSmiConstant(0, mode),
      TaggedToParameter(length, mode), nullptr,
      ExtractFixedArrayFlag::kAllFixedArraysDontCopyCOW, mode);

  // Use the cannonical map for the Array's ElementsKind
  Node* native_context = LoadNativeContext(context);
  Node* array_map = LoadJSArrayElementsMap(elements_kind, native_context);

  Node* result = AllocateUninitializedJSArrayWithoutElements(array_map, length,
                                                             allocation_site);
  StoreObjectField(result, JSObject::kElementsOffset, new_elements);
  return result;
}
