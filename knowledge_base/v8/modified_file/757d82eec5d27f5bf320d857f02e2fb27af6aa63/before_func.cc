Node* CodeStubAssembler::AllocateRawUnaligned(Node* size_in_bytes,
                                              AllocationFlags flags,
                                              Node* top_address,
                                              Node* limit_address) {
  Node* top = Load(MachineType::Pointer(), top_address);
  Node* limit = Load(MachineType::Pointer(), limit_address);

  // If there's not enough space, call the runtime.
  Variable result(this, MachineRepresentation::kTagged);
  Label runtime_call(this, Label::kDeferred), no_runtime_call(this);
  Label merge_runtime(this, &result);

  Branch(IntPtrLessThan(IntPtrSub(limit, top), size_in_bytes), &runtime_call,
         &no_runtime_call);

  Bind(&runtime_call);
  // AllocateInTargetSpace does not use the context.
  Node* context = IntPtrConstant(0);
  Node* runtime_flags = SmiTag(Int32Constant(
      AllocateDoubleAlignFlag::encode(false) |
      AllocateTargetSpace::encode(flags & kPretenured
                                      ? AllocationSpace::OLD_SPACE
                                      : AllocationSpace::NEW_SPACE)));
  Node* runtime_result = CallRuntime(Runtime::kAllocateInTargetSpace, context,
                                     SmiTag(size_in_bytes), runtime_flags);
  result.Bind(runtime_result);
  Goto(&merge_runtime);

  // When there is enough space, return `top' and bump it up.
  Bind(&no_runtime_call);
  Node* no_runtime_result = top;
  StoreNoWriteBarrier(MachineType::PointerRepresentation(), top_address,
                      IntPtrAdd(top, size_in_bytes));
  no_runtime_result = BitcastWordToTagged(
      IntPtrAdd(no_runtime_result, IntPtrConstant(kHeapObjectTag)));
  result.Bind(no_runtime_result);
  Goto(&merge_runtime);

  Bind(&merge_runtime);
  return result.value();
}
