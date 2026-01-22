void WasmGraphBuilder::StackCheck(
    WasmInstanceCacheNodes* shared_memory_instance_cache,
    wasm::WasmCodePosition position) {
  DCHECK_NOT_NULL(env_);  // Wrappers don't get stack checks.
  if (!FLAG_wasm_stack_checks || !env_->runtime_exception_support) {
    return;
  }

  Node* limit_address =
      LOAD_INSTANCE_FIELD(StackLimitAddress, MachineType::Pointer());
  Node* limit = gasm_->LoadFromObject(MachineType::Pointer(), limit_address, 0);

  Node* check = SetEffect(graph()->NewNode(
      mcgraph()->machine()->StackPointerGreaterThan(StackCheckKind::kWasm),
      limit, effect()));

  Node* if_true;
  Node* if_false;
  BranchExpectTrue(check, &if_true, &if_false);

  if (stack_check_call_operator_ == nullptr) {
    // Build and cache the stack check call operator and the constant
    // representing the stack check code.

    // A direct call to a wasm runtime stub defined in this module.
    // Just encode the stub index. This will be patched at relocation.
    stack_check_code_node_.set(mcgraph()->RelocatableIntPtrConstant(
        wasm::WasmCode::kWasmStackGuard, RelocInfo::WASM_STUB_CALL));

    auto call_descriptor = Linkage::GetStubCallDescriptor(
        mcgraph()->zone(),                    // zone
        NoContextDescriptor{},                // descriptor
        0,                                    // stack parameter count
        CallDescriptor::kNoFlags,             // flags
        Operator::kNoThrow,                   // properties
        StubCallMode::kCallWasmRuntimeStub);  // stub call mode
    stack_check_call_operator_ = mcgraph()->common()->Call(call_descriptor);
  }

  Node* call =
      graph()->NewNode(stack_check_call_operator_.get(),
                       stack_check_code_node_.get(), effect(), if_false);
  SetSourcePosition(call, position);

  DCHECK_GT(call->op()->EffectOutputCount(), 0);
  DCHECK_EQ(call->op()->ControlOutputCount(), 0);

  SetEffectControl(call, if_false);

  Node* merge = Merge(if_true, control());
  Node* ephi_inputs[] = {check, effect(), merge};
  Node* ephi = EffectPhi(2, ephi_inputs);

  // We only need to refresh the size of a shared memory, as its start can never
  // change.
  if (shared_memory_instance_cache != nullptr) {
    Node* new_memory_size =
        LOAD_MUTABLE_INSTANCE_FIELD(MemorySize, MachineType::UintPtr());
    shared_memory_instance_cache->mem_size = CreateOrMergeIntoPhi(
        MachineType::PointerRepresentation(), merge,
        shared_memory_instance_cache->mem_size, new_memory_size);
  }

  SetEffectControl(ephi, merge);
}
