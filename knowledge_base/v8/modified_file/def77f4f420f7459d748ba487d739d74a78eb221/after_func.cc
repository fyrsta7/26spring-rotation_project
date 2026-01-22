JSNativeContextSpecialization::ValueEffectControl
JSNativeContextSpecialization::BuildPropertyAccess(
    Node* receiver, Node* value, Node* context, Node* frame_state, Node* effect,
    Node* control, Handle<Name> name, Handle<Context> native_context,
    PropertyAccessInfo const& access_info, AccessMode access_mode) {
  // Determine actual holder and perform prototype chain checks.
  Handle<JSObject> holder;
  if (access_info.holder().ToHandle(&holder)) {
    AssumePrototypesStable(access_info.receiver_maps(), native_context, holder);
  }

  // Generate the actual property access.
  if (access_info.IsNotFound()) {
    DCHECK_EQ(AccessMode::kLoad, access_mode);
    value = jsgraph()->UndefinedConstant();
  } else if (access_info.IsDataConstant()) {
    value = jsgraph()->Constant(access_info.constant());
    if (access_mode == AccessMode::kStore) {
      Node* check =
          graph()->NewNode(simplified()->ReferenceEqual(), value, value);
      effect =
          graph()->NewNode(simplified()->CheckIf(), check, effect, control);
    }
  } else if (access_info.IsAccessorConstant()) {
    // TODO(bmeurer): Properly rewire the IfException edge here if there's any.
    Node* target = jsgraph()->Constant(access_info.constant());
    FrameStateInfo const& frame_info = OpParameter<FrameStateInfo>(frame_state);
    Handle<SharedFunctionInfo> shared_info =
        frame_info.shared_info().ToHandleChecked();
    switch (access_mode) {
      case AccessMode::kLoad: {
        // We need a FrameState for the getter stub to restore the correct
        // context before returning to fullcodegen.
        FrameStateFunctionInfo const* frame_info0 =
            common()->CreateFrameStateFunctionInfo(FrameStateType::kGetterStub,
                                                   1, 0, shared_info);
        Node* frame_state0 = graph()->NewNode(
            common()->FrameState(BailoutId::None(),
                                 OutputFrameStateCombine::Ignore(),
                                 frame_info0),
            graph()->NewNode(common()->StateValues(1), receiver),
            jsgraph()->EmptyStateValues(), jsgraph()->EmptyStateValues(),
            context, target, frame_state);

        // Introduce the call to the getter function.
        value = effect = graph()->NewNode(
            javascript()->CallFunction(
                2, 0.0f, VectorSlotPair(),
                ConvertReceiverMode::kNotNullOrUndefined),
            target, receiver, context, frame_state0, effect, control);
        control = graph()->NewNode(common()->IfSuccess(), value);
        break;
      }
      case AccessMode::kStore: {
        // We need a FrameState for the setter stub to restore the correct
        // context and return the appropriate value to fullcodegen.
        FrameStateFunctionInfo const* frame_info0 =
            common()->CreateFrameStateFunctionInfo(FrameStateType::kSetterStub,
                                                   2, 0, shared_info);
        Node* frame_state0 = graph()->NewNode(
            common()->FrameState(BailoutId::None(),
                                 OutputFrameStateCombine::Ignore(),
                                 frame_info0),
            graph()->NewNode(common()->StateValues(2), receiver, value),
            jsgraph()->EmptyStateValues(), jsgraph()->EmptyStateValues(),
            context, target, frame_state);

        // Introduce the call to the setter function.
        effect = graph()->NewNode(javascript()->CallFunction(
                                      3, 0.0f, VectorSlotPair(),
                                      ConvertReceiverMode::kNotNullOrUndefined),
                                  target, receiver, value, context,
                                  frame_state0, effect, control);
        control = graph()->NewNode(common()->IfSuccess(), effect);
        break;
      }
    }
  } else {
    DCHECK(access_info.IsDataField());
    FieldIndex const field_index = access_info.field_index();
    Type* const field_type = access_info.field_type();
    MachineRepresentation const field_representation =
        access_info.field_representation();
    if (access_mode == AccessMode::kLoad) {
      if (access_info.holder().ToHandle(&holder)) {
        receiver = jsgraph()->Constant(holder);
      }
      // Optimize immutable property loads.
      HeapObjectMatcher m(receiver);
      if (m.HasValue() && m.Value()->IsJSObject()) {
        // TODO(turbofan): Given that we already have the field_index here, we
        // might be smarter in the future and not rely on the LookupIterator,
        // but for now let's just do what Crankshaft does.
        LookupIterator it(m.Value(), name,
                          LookupIterator::OWN_SKIP_INTERCEPTOR);
        if (it.IsFound() && it.IsReadOnly() && !it.IsConfigurable()) {
          Node* value = jsgraph()->Constant(JSReceiver::GetDataProperty(&it));
          return ValueEffectControl(value, effect, control);
        }
      }
    }
    Node* storage = receiver;
    if (!field_index.is_inobject()) {
      storage = effect = graph()->NewNode(
          simplified()->LoadField(AccessBuilder::ForJSObjectProperties()),
          storage, effect, control);
    }
    FieldAccess field_access = {
        kTaggedBase,
        field_index.offset(),
        name,
        field_type,
        MachineType::TypeForRepresentation(field_representation),
        kFullWriteBarrier};
    if (access_mode == AccessMode::kLoad) {
      if (field_representation == MachineRepresentation::kFloat64) {
        if (!field_index.is_inobject() || field_index.is_hidden_field() ||
            !FLAG_unbox_double_fields) {
          FieldAccess const storage_access = {kTaggedBase,
                                              field_index.offset(),
                                              name,
                                              Type::OtherInternal(),
                                              MachineType::TaggedPointer(),
                                              kPointerWriteBarrier};
          storage = effect =
              graph()->NewNode(simplified()->LoadField(storage_access), storage,
                               effect, control);
          field_access.offset = HeapNumber::kValueOffset;
          field_access.name = MaybeHandle<Name>();
        }
      }
      // TODO(turbofan): Track the field_map (if any) on the {field_access} and
      // use it in LoadElimination to eliminate map checks.
      value = effect = graph()->NewNode(simplified()->LoadField(field_access),
                                        storage, effect, control);
    } else {
      DCHECK_EQ(AccessMode::kStore, access_mode);
      switch (field_representation) {
        case MachineRepresentation::kFloat64: {
          value = effect = graph()->NewNode(simplified()->CheckNumber(), value,
                                            effect, control);
          if (!field_index.is_inobject() || field_index.is_hidden_field() ||
              !FLAG_unbox_double_fields) {
            if (access_info.HasTransitionMap()) {
              // Allocate a MutableHeapNumber for the new property.
              effect = graph()->NewNode(
                  common()->BeginRegion(RegionObservability::kNotObservable),
                  effect);
              Node* box = effect = graph()->NewNode(
                  simplified()->Allocate(NOT_TENURED),
                  jsgraph()->Constant(HeapNumber::kSize), effect, control);
              effect = graph()->NewNode(
                  simplified()->StoreField(AccessBuilder::ForMap()), box,
                  jsgraph()->HeapConstant(factory()->mutable_heap_number_map()),
                  effect, control);
              effect = graph()->NewNode(
                  simplified()->StoreField(AccessBuilder::ForHeapNumberValue()),
                  box, value, effect, control);
              value = effect =
                  graph()->NewNode(common()->FinishRegion(), box, effect);

              field_access.type = Type::Any();
              field_access.machine_type = MachineType::TaggedPointer();
              field_access.write_barrier_kind = kPointerWriteBarrier;
            } else {
              // We just store directly to the MutableHeapNumber.
              FieldAccess const storage_access = {kTaggedBase,
                                                  field_index.offset(),
                                                  name,
                                                  Type::OtherInternal(),
                                                  MachineType::TaggedPointer(),
                                                  kPointerWriteBarrier};
              storage = effect =
                  graph()->NewNode(simplified()->LoadField(storage_access),
                                   storage, effect, control);
              field_access.offset = HeapNumber::kValueOffset;
              field_access.name = MaybeHandle<Name>();
              field_access.machine_type = MachineType::Float64();
            }
          }
          break;
        }
        case MachineRepresentation::kTaggedSigned: {
          value = effect = graph()->NewNode(simplified()->CheckSmi(), value,
                                            effect, control);
          field_access.write_barrier_kind = kNoWriteBarrier;
          break;
        }
        case MachineRepresentation::kTaggedPointer: {
          // Ensure that {value} is a HeapObject.
          value = effect = graph()->NewNode(simplified()->CheckHeapObject(),
                                            value, effect, control);
          Handle<Map> field_map;
          if (access_info.field_map().ToHandle(&field_map)) {
            // Emit a map check for the value.
            effect = graph()->NewNode(simplified()->CheckMaps(1), value,
                                      jsgraph()->HeapConstant(field_map),
                                      effect, control);
          }
          field_access.write_barrier_kind = kPointerWriteBarrier;
          break;
        }
        case MachineRepresentation::kTagged:
          break;
        case MachineRepresentation::kNone:
        case MachineRepresentation::kBit:
        case MachineRepresentation::kWord8:
        case MachineRepresentation::kWord16:
        case MachineRepresentation::kWord32:
        case MachineRepresentation::kWord64:
        case MachineRepresentation::kFloat32:
        case MachineRepresentation::kSimd128:
          UNREACHABLE();
          break;
      }
      Handle<Map> transition_map;
      if (access_info.transition_map().ToHandle(&transition_map)) {
        effect = graph()->NewNode(
            common()->BeginRegion(RegionObservability::kObservable), effect);
        effect = graph()->NewNode(
            simplified()->StoreField(AccessBuilder::ForMap()), receiver,
            jsgraph()->Constant(transition_map), effect, control);
      }
      effect = graph()->NewNode(simplified()->StoreField(field_access), storage,
                                value, effect, control);
      if (access_info.HasTransitionMap()) {
        effect = graph()->NewNode(common()->FinishRegion(),
                                  jsgraph()->UndefinedConstant(), effect);
      }
    }
  }

  return ValueEffectControl(value, effect, control);
}
