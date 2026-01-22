void AsyncBuiltinsAssembler::InitializeNativeClosure(Node* context,
                                                     Node* native_context,
                                                     Node* function,
                                                     Node* context_index) {
  Node* const function_map = LoadContextElement(
      native_context, Context::STRICT_FUNCTION_WITHOUT_PROTOTYPE_MAP_INDEX);
  // Ensure that we don't have to initialize prototype_or_initial_map field of
  // JSFunction.
  CSA_ASSERT(this, WordEqual(LoadMapInstanceSizeInWords(function_map),
                             IntPtrConstant(JSFunction::kSizeWithoutPrototype /
                                            kPointerSize)));
  STATIC_ASSERT(JSFunction::kSizeWithoutPrototype == 7 * kPointerSize);
  StoreMapNoWriteBarrier(function, function_map);
  StoreObjectFieldRoot(function, JSObject::kPropertiesOrHashOffset,
                       RootIndex::kEmptyFixedArray);
  StoreObjectFieldRoot(function, JSObject::kElementsOffset,
                       RootIndex::kEmptyFixedArray);
  StoreObjectFieldRoot(function, JSFunction::kFeedbackCellOffset,
                       RootIndex::kManyClosuresCell);

  Node* shared_info = LoadContextElement(native_context, context_index);
  CSA_ASSERT(this, IsSharedFunctionInfo(shared_info));
  StoreObjectFieldNoWriteBarrier(
      function, JSFunction::kSharedFunctionInfoOffset, shared_info);
  StoreObjectFieldNoWriteBarrier(function, JSFunction::kContextOffset, context);

  Node* const code = GetSharedFunctionInfoCode(shared_info);
  StoreObjectFieldNoWriteBarrier(function, JSFunction::kCodeOffset, code);
}
