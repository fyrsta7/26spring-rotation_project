void AsyncBuiltinsAssembler::InitializeNativeClosure(Node* context,
                                                     Node* native_context,
                                                     Node* function,
                                                     Node* context_index) {
  TNode<Map> function_map = CAST(LoadContextElement(
      native_context, Context::STRICT_FUNCTION_WITHOUT_PROTOTYPE_MAP_INDEX));
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

  TNode<SharedFunctionInfo> shared_info =
      CAST(LoadContextElement(native_context, context_index));
  StoreObjectFieldNoWriteBarrier(
      function, JSFunction::kSharedFunctionInfoOffset, shared_info);
  StoreObjectFieldNoWriteBarrier(function, JSFunction::kContextOffset, context);

  // For the native closures that are initialized here (for `await`)
  // we know that their SharedFunctionInfo::function_data() slot
  // contains a builtin index (as Smi), so there's no need to use
  // CodeStubAssembler::GetSharedFunctionInfoCode() helper here,
  // which almost doubles the size of `await` builtins (unnecessarily).
  TNode<Smi> builtin_id = LoadObjectField<Smi>(
      shared_info, SharedFunctionInfo::kFunctionDataOffset);
  TNode<Code> code = LoadBuiltin(builtin_id);
  StoreObjectFieldNoWriteBarrier(function, JSFunction::kCodeOffset, code);
}
