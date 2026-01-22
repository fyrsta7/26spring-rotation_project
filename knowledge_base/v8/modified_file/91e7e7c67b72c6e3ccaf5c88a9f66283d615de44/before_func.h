  OpIndex REDUCE(Allocate)(OpIndex size, AllocationType type) {
    DCHECK_EQ(type, any_of(AllocationType::kYoung, AllocationType::kOld));

    if (v8_flags.single_generation && type == AllocationType::kYoung) {
      type = AllocationType::kOld;
    }

    OpIndex top_address;
    if (isolate_ != nullptr) {
      top_address = __ ExternalConstant(
          type == AllocationType::kYoung
              ? ExternalReference::new_space_allocation_top_address(isolate_)
              : ExternalReference::old_space_allocation_top_address(isolate_));
    } else {
      // Wasm mode: producing isolate-independent code, loading the isolate
      // address at runtime.
#if V8_ENABLE_WEBASSEMBLY
      V<WasmInstanceObject> instance_node = __ WasmInstanceParameter();
      int top_address_offset =
          type == AllocationType::kYoung
              ? WasmInstanceObject::kNewAllocationTopAddressOffset
              : WasmInstanceObject::kOldAllocationTopAddressOffset;
      top_address =
          __ Load(instance_node, LoadOp::Kind::TaggedBase().Immutable(),
                  MemoryRepresentation::PointerSized(), top_address_offset);
#else
      UNREACHABLE();
#endif  // V8_ENABLE_WEBASSEMBLY
    }

    if (analyzer_->IsFoldedAllocation(__ current_operation_origin())) {
      DCHECK_NE(__ GetVariable(top(type)), OpIndex::Invalid());
      OpIndex obj_addr = __ GetVariable(top(type));
      __ SetVariable(top(type), __ PointerAdd(__ GetVariable(top(type)), size));
      __ StoreOffHeap(top_address, __ GetVariable(top(type)),
                      MemoryRepresentation::PointerSized());
      return __ BitcastWordPtrToTagged(
          __ PointerAdd(obj_addr, __ IntPtrConstant(kHeapObjectTag)));
    }

    __ SetVariable(
        top(type),
        __ LoadOffHeap(top_address, MemoryRepresentation::PointerSized()));

    OpIndex allocate_builtin;
    if (isolate_ != nullptr) {
      if (type == AllocationType::kYoung) {
        allocate_builtin =
            __ BuiltinCode(Builtin::kAllocateInYoungGeneration, isolate_);
      } else {
        allocate_builtin =
            __ BuiltinCode(Builtin::kAllocateInOldGeneration, isolate_);
      }
    } else {
      // This lowering is used by Wasm, where we compile isolate-independent
      // code. Builtin calls simply encode the target builtin ID, which will
      // be patched to the builtin's address later.
#if V8_ENABLE_WEBASSEMBLY
      Builtin builtin;
      if (type == AllocationType::kYoung) {
        builtin = Builtin::kAllocateInYoungGeneration;
      } else {
        builtin = Builtin::kAllocateInOldGeneration;
      }
      static_assert(std::is_same<Smi, BuiltinPtr>(), "BuiltinPtr must be Smi");
      allocate_builtin = __ NumberConstant(static_cast<int>(builtin));
#else
      UNREACHABLE();
#endif
    }

    Block* call_runtime = __ NewBlock();
    Block* done = __ NewBlock();

    OpIndex limit_address = GetLimitAddress(type);
    OpIndex limit =
        __ LoadOffHeap(limit_address, MemoryRepresentation::PointerSized());

    // If the allocation size is not statically known or is known to be larger
    // than kMaxRegularHeapObjectSize, do not update {top(type)} in case of a
    // runtime call. This is needed because we cannot allocation-fold large and
    // normal-sized objects.
    uint64_t constant_size{};
    if (!__ matcher().MatchIntegralWordConstant(
            size, WordRepresentation::PointerSized(), &constant_size) ||
        constant_size > kMaxRegularHeapObjectSize) {
      Variable result =
          __ NewLoopInvariantVariable(RegisterRepresentation::Tagged());
      if (!constant_size) {
        // Check if we can do bump pointer allocation here.
        OpIndex top_value = __ GetVariable(top(type));
        __ SetVariable(result,
                       __ BitcastWordPtrToTagged(__ WordPtrAdd(
                           top_value, __ IntPtrConstant(kHeapObjectTag))));
        OpIndex new_top = __ PointerAdd(top_value, size);
        __ GotoIfNot(LIKELY(__ UintPtrLessThan(new_top, limit)), call_runtime);
        __ GotoIfNot(LIKELY(__ UintPtrLessThan(
                         size, __ IntPtrConstant(kMaxRegularHeapObjectSize))),
                     call_runtime);
        __ SetVariable(top(type), new_top);
        __ StoreOffHeap(top_address, new_top,
                        MemoryRepresentation::PointerSized());
        __ Goto(done);
      }
      if (constant_size || __ Bind(call_runtime)) {
        __ SetVariable(result, __ Call(allocate_builtin, {size},
                                       AllocateBuiltinDescriptor()));
        __ Goto(done);
      }

      __ BindReachable(done);
      return __ GetVariable(result);
    }

    OpIndex reservation_size;
    if (auto c = analyzer_->ReservedSize(__ current_operation_origin())) {
      reservation_size = __ UintPtrConstant(*c);
    } else {
      reservation_size = size;
    }
    // Check if we can do bump pointer allocation here.
    bool reachable =
        __ GotoIfNot(__ UintPtrLessThan(
                         size, __ IntPtrConstant(kMaxRegularHeapObjectSize)),
                     call_runtime, BranchHint::kTrue) !=
        ConditionalGotoStatus::kGotoDestination;
    if (reachable) {
      __ Branch(__ UintPtrLessThan(
                    __ PointerAdd(__ GetVariable(top(type)), reservation_size),
                    limit),
                done, call_runtime, BranchHint::kTrue);
    }

    // Call the runtime if bump pointer area exhausted.
    if (__ Bind(call_runtime)) {
      OpIndex allocated = __ Call(allocate_builtin, {reservation_size},
                                  AllocateBuiltinDescriptor());
      __ SetVariable(top(type),
                     __ PointerSub(__ BitcastTaggedToWord(allocated),
                                   __ IntPtrConstant(kHeapObjectTag)));
      __ Goto(done);
    }

    __ BindReachable(done);
    // Compute the new top and write it back.
    OpIndex obj_addr = __ GetVariable(top(type));
    __ SetVariable(top(type), __ PointerAdd(__ GetVariable(top(type)), size));
    __ StoreOffHeap(top_address, __ GetVariable(top(type)),
                    MemoryRepresentation::PointerSized());
    return __ BitcastWordPtrToTagged(
        __ PointerAdd(obj_addr, __ IntPtrConstant(kHeapObjectTag)));
  }
