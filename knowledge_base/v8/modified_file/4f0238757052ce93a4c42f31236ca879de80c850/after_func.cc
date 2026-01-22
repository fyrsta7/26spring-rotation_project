Node* AsyncBuiltinsAssembler::AwaitOld(Node* context, Node* generator,
                                       Node* value, Node* outer_promise,
                                       Node* on_resolve_context_index,
                                       Node* on_reject_context_index,
                                       Node* is_predicted_as_caught) {
  Node* const native_context = LoadNativeContext(context);

  static const int kWrappedPromiseOffset =
      FixedArray::SizeFor(Context::MIN_CONTEXT_SLOTS);
  static const int kResolveClosureOffset =
      kWrappedPromiseOffset + JSPromise::kSizeWithEmbedderFields;
  static const int kRejectClosureOffset =
      kResolveClosureOffset + JSFunction::kSizeWithoutPrototype;
  static const int kTotalSize =
      kRejectClosureOffset + JSFunction::kSizeWithoutPrototype;

  Node* const base = AllocateInNewSpace(kTotalSize);
  Node* const closure_context = base;
  {
    // Initialize the await context, storing the {generator} as extension.
    StoreMapNoWriteBarrier(closure_context, RootIndex::kAwaitContextMap);
    StoreObjectFieldNoWriteBarrier(closure_context, Context::kLengthOffset,
                                   SmiConstant(Context::MIN_CONTEXT_SLOTS));
    Node* const empty_scope_info =
        LoadContextElement(native_context, Context::SCOPE_INFO_INDEX);
    StoreContextElementNoWriteBarrier(
        closure_context, Context::SCOPE_INFO_INDEX, empty_scope_info);
    StoreContextElementNoWriteBarrier(closure_context, Context::PREVIOUS_INDEX,
                                      native_context);
    StoreContextElementNoWriteBarrier(closure_context, Context::EXTENSION_INDEX,
                                      generator);
    StoreContextElementNoWriteBarrier(
        closure_context, Context::NATIVE_CONTEXT_INDEX, native_context);
  }

  // Let promiseCapability be ! NewPromiseCapability(%Promise%).
  Node* const promise_fun =
      LoadContextElement(native_context, Context::PROMISE_FUNCTION_INDEX);
  CSA_ASSERT(this, IsFunctionWithPrototypeSlotMap(LoadMap(promise_fun)));
  Node* const promise_map =
      LoadObjectField(promise_fun, JSFunction::kPrototypeOrInitialMapOffset);
  // Assert that the JSPromise map has an instance size is
  // JSPromise::kSizeWithEmbedderFields.
  CSA_ASSERT(this, WordEqual(LoadMapInstanceSizeInWords(promise_map),
                             IntPtrConstant(JSPromise::kSizeWithEmbedderFields /
                                            kPointerSize)));
  Node* const wrapped_value = InnerAllocate(base, kWrappedPromiseOffset);
  {
    // Initialize Promise
    StoreMapNoWriteBarrier(wrapped_value, promise_map);
    StoreObjectFieldRoot(wrapped_value, JSPromise::kPropertiesOrHashOffset,
                         RootIndex::kEmptyFixedArray);
    StoreObjectFieldRoot(wrapped_value, JSPromise::kElementsOffset,
                         RootIndex::kEmptyFixedArray);
    PromiseInit(wrapped_value);
  }

  // Initialize resolve handler
  Node* const on_resolve = InnerAllocate(base, kResolveClosureOffset);
  InitializeNativeClosure(closure_context, native_context, on_resolve,
                          on_resolve_context_index);

  // Initialize reject handler
  Node* const on_reject = InnerAllocate(base, kRejectClosureOffset);
  InitializeNativeClosure(closure_context, native_context, on_reject,
                          on_reject_context_index);

  VARIABLE(var_throwaway, MachineRepresentation::kTaggedPointer,
           UndefinedConstant());

  // Deal with PromiseHooks and debug support in the runtime. This
  // also allocates the throwaway promise, which is only needed in
  // case of PromiseHooks or debugging.
  Label if_debugging(this, Label::kDeferred), do_resolve_promise(this);
  GotoIf(IsDebugActive(), &if_debugging);
  Branch(IsPromiseHookEnabledOrHasAsyncEventDelegate(), &if_debugging,
         &do_resolve_promise);
  BIND(&if_debugging);
  var_throwaway.Bind(CallRuntime(Runtime::kAwaitPromisesInitOld, context, value,
                                 wrapped_value, outer_promise, on_reject,
                                 is_predicted_as_caught));
  Goto(&do_resolve_promise);
  BIND(&do_resolve_promise);

  // Perform ! Call(promiseCapability.[[Resolve]], undefined, « promise »).
  CallBuiltin(Builtins::kResolvePromise, context, wrapped_value, value);

  return CallBuiltin(Builtins::kPerformPromiseThen, context, wrapped_value,
                     on_resolve, on_reject, var_throwaway.value());
}
