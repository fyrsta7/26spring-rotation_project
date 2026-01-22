Node* CodeStubAssembler::CloneFastJSArray(Node* context, Node* array,
                                          ParameterMode mode,
                                          Node* allocation_site) {
  Node* length = LoadJSArrayLength(array);
  Node* elements = LoadElements(array);

  Node* original_array_map = LoadMap(array);
  Node* elements_kind = LoadMapElementsKind(original_array_map);

  Node* new_elements = CloneFixedArray(elements);

  // Use the cannonical map for the Array's ElementsKind
  Node* native_context = LoadNativeContext(context);
  Node* array_map = LoadJSArrayElementsMap(elements_kind, native_context);
  Node* result = AllocateUninitializedJSArrayWithoutElements(array_map, length,
                                                             allocation_site);
  StoreObjectField(result, JSObject::kElementsOffset, new_elements);
  return result;
}
