Reduction JSCallReducer::ReduceArrayIteratorPrototypeNext(Node* node) {
  DCHECK_EQ(IrOpcode::kJSCall, node->opcode());
  CallParameters const& p = CallParametersOf(node->op());
  Node* iterator = NodeProperties::GetValueInput(node, 1);
  Node* context = NodeProperties::GetContextInput(node);
  Node* effect = NodeProperties::GetEffectInput(node);
  Node* control = NodeProperties::GetControlInput(node);

  if (p.speculation_mode() == SpeculationMode::kDisallowSpeculation) {
    return NoChange();
  }

  // Check if the {iterator} is a JSCreateArrayIterator.
  if (iterator->opcode() != IrOpcode::kJSCreateArrayIterator) return NoChange();
  IterationKind const iteration_kind =
      CreateArrayIteratorParametersOf(iterator->op()).kind();

  // Try to infer the [[IteratedObject]] maps from the {iterator}.
  ZoneHandleSet<Map> iterated_object_maps;
  if (!InferIteratedObjectMaps(isolate(), iterator, &iterated_object_maps)) {
    return NoChange();
  }
  DCHECK_NE(0, iterated_object_maps.size());

  // Check that various {iterated_object_maps} have compatible elements kinds.
  ElementsKind elements_kind = iterated_object_maps[0]->elements_kind();
  if (IsFixedTypedArrayElementsKind(elements_kind)) {
    // TurboFan doesn't support loading from BigInt typed arrays yet.
    if (elements_kind == BIGUINT64_ELEMENTS ||
        elements_kind == BIGINT64_ELEMENTS) {
      return NoChange();
    }
    for (Handle<Map> iterated_object_map : iterated_object_maps) {
      if (iterated_object_map->elements_kind() != elements_kind) {
        return NoChange();
      }
    }
  } else {
    for (Handle<Map> iterated_object_map : iterated_object_maps) {
      if (!CanInlineArrayIteratingBuiltin(isolate(), iterated_object_map)) {
        return NoChange();
      }
      if (!UnionElementsKindUptoSize(&elements_kind,
                                     iterated_object_map->elements_kind())) {
        return NoChange();
      }
    }
  }

  // Install code dependency on the array protector for holey arrays.
  if (IsHoleyElementsKind(elements_kind)) {
    dependencies()->DependOnProtector(
        PropertyCellRef(js_heap_broker(), factory()->no_elements_protector()));
  }

  // Load the (current) {iterated_object} from the {iterator}.
  Node* iterated_object = effect =
      graph()->NewNode(simplified()->LoadField(
                           AccessBuilder::ForJSArrayIteratorIteratedObject()),
                       iterator, effect, control);

  // Ensure that the {iterated_object} map didn't change.
  effect = graph()->NewNode(
      simplified()->CheckMaps(CheckMapsFlag::kNone, iterated_object_maps,
                              p.feedback()),
      iterated_object, effect, control);

  if (IsFixedTypedArrayElementsKind(elements_kind)) {
    // See if we can skip the neutering check.
    if (isolate()->IsArrayBufferNeuteringIntact()) {
      // Add a code dependency so we are deoptimized in case an ArrayBuffer
      // gets neutered.
      dependencies()->DependOnProtector(PropertyCellRef(
          js_heap_broker(), factory()->array_buffer_neutering_protector()));
    } else {
      // Deoptimize if the array buffer was neutered.
      Node* buffer = effect = graph()->NewNode(
          simplified()->LoadField(AccessBuilder::ForJSArrayBufferViewBuffer()),
          iterated_object, effect, control);

      Node* check = effect = graph()->NewNode(
          simplified()->ArrayBufferWasNeutered(), buffer, effect, control);
      check = graph()->NewNode(simplified()->BooleanNot(), check);
      // TODO(bmeurer): Pass p.feedback(), or better introduce
      // CheckArrayBufferNotNeutered?
      effect = graph()->NewNode(
          simplified()->CheckIf(DeoptimizeReason::kArrayBufferWasNeutered),
          check, effect, control);
    }
  }

  // Load the [[NextIndex]] from the {iterator} and leverage the fact
  // that we definitely know that it's in Unsigned32 range since the
  // {iterated_object} is either a JSArray or a JSTypedArray. For the
  // latter case we even know that it's a Smi in UnsignedSmall range.
  FieldAccess index_access = AccessBuilder::ForJSArrayIteratorNextIndex();
  if (IsFixedTypedArrayElementsKind(elements_kind)) {
    index_access.type = TypeCache::Get().kJSTypedArrayLengthType;
    index_access.machine_type = MachineType::TaggedSigned();
    index_access.write_barrier_kind = kNoWriteBarrier;
  } else {
    index_access.type = TypeCache::Get().kJSArrayLengthType;
  }
  Node* index = effect = graph()->NewNode(simplified()->LoadField(index_access),
                                          iterator, effect, control);

  // Load the elements of the {iterated_object}. While it feels
  // counter-intuitive to place the elements pointer load before
  // the condition below, as it might not be needed (if the {index}
  // is out of bounds for the {iterated_object}), it's better this
  // way as it allows the LoadElimination to eliminate redundant
  // reloads of the elements pointer.
  Node* elements = effect = graph()->NewNode(
      simplified()->LoadField(AccessBuilder::ForJSObjectElements()),
      iterated_object, effect, control);

  // Load the length of the {iterated_object}. Due to the map checks we
  // already know something about the length here, which we can leverage
  // to generate Word32 operations below without additional checking.
  FieldAccess length_access =
      IsFixedTypedArrayElementsKind(elements_kind)
          ? AccessBuilder::ForJSTypedArrayLength()
          : AccessBuilder::ForJSArrayLength(elements_kind);
  Node* length = effect = graph()->NewNode(
      simplified()->LoadField(length_access), iterated_object, effect, control);

  // Check whether {index} is within the valid range for the {iterated_object}.
  Node* check = graph()->NewNode(simplified()->NumberLessThan(), index, length);
  Node* branch =
      graph()->NewNode(common()->Branch(BranchHint::kTrue), check, control);

  Node* done_true;
  Node* value_true;
  Node* etrue = effect;
  Node* if_true = graph()->NewNode(common()->IfTrue(), branch);
  {
    // We know that the {index} is range of the {length} now.
    index = etrue = graph()->NewNode(
        common()->TypeGuard(
            Type::Range(0.0, length_access.type.Max() - 1.0, graph()->zone())),
        index, etrue, if_true);

    done_true = jsgraph()->FalseConstant();
    if (iteration_kind == IterationKind::kKeys) {
      // Just return the {index}.
      value_true = index;
    } else {
      DCHECK(iteration_kind == IterationKind::kEntries ||
             iteration_kind == IterationKind::kValues);

      if (IsFixedTypedArrayElementsKind(elements_kind)) {
        Node* base_ptr = etrue = graph()->NewNode(
            simplified()->LoadField(
                AccessBuilder::ForFixedTypedArrayBaseBasePointer()),
            elements, etrue, if_true);
        Node* external_ptr = etrue = graph()->NewNode(
            simplified()->LoadField(
                AccessBuilder::ForFixedTypedArrayBaseExternalPointer()),
            elements, etrue, if_true);

        ExternalArrayType array_type = kExternalInt8Array;
        switch (elements_kind) {
#define TYPED_ARRAY_CASE(Type, type, TYPE, ctype) \
  case TYPE##_ELEMENTS:                           \
    array_type = kExternal##Type##Array;          \
    break;
          TYPED_ARRAYS(TYPED_ARRAY_CASE)
          default:
            UNREACHABLE();
#undef TYPED_ARRAY_CASE
        }

        Node* buffer = etrue =
            graph()->NewNode(simplified()->LoadField(
                                 AccessBuilder::ForJSArrayBufferViewBuffer()),
                             iterated_object, etrue, if_true);

        value_true = etrue =
            graph()->NewNode(simplified()->LoadTypedElement(array_type), buffer,
                             base_ptr, external_ptr, index, etrue, if_true);
      } else {
        value_true = etrue = graph()->NewNode(
            simplified()->LoadElement(
                AccessBuilder::ForFixedArrayElement(elements_kind)),
            elements, index, etrue, if_true);

        // Convert hole to undefined if needed.
        if (elements_kind == HOLEY_ELEMENTS ||
            elements_kind == HOLEY_SMI_ELEMENTS) {
          value_true = graph()->NewNode(
              simplified()->ConvertTaggedHoleToUndefined(), value_true);
        } else if (elements_kind == HOLEY_DOUBLE_ELEMENTS) {
          // TODO(6587): avoid deopt if not all uses of value are truncated.
          CheckFloat64HoleMode mode = CheckFloat64HoleMode::kAllowReturnHole;
          value_true = etrue = graph()->NewNode(
              simplified()->CheckFloat64Hole(mode, p.feedback()), value_true,
              etrue, if_true);
        }
      }

      if (iteration_kind == IterationKind::kEntries) {
        // Allocate elements for key/value pair
        value_true = etrue =
            graph()->NewNode(javascript()->CreateKeyValueArray(), index,
                             value_true, context, etrue);
      } else {
        DCHECK_EQ(IterationKind::kValues, iteration_kind);
      }
    }

    // Increment the [[NextIndex]] field in the {iterator}. The TypeGuards
    // above guarantee that the {next_index} is in the UnsignedSmall range.
    Node* next_index = graph()->NewNode(simplified()->NumberAdd(), index,
                                        jsgraph()->OneConstant());
    etrue = graph()->NewNode(simplified()->StoreField(index_access), iterator,
                             next_index, etrue, if_true);
  }

  Node* done_false;
  Node* value_false;
  Node* efalse = effect;
  Node* if_false = graph()->NewNode(common()->IfFalse(), branch);
  {
    // iterator.[[NextIndex]] >= array.length, stop iterating.
    done_false = jsgraph()->TrueConstant();
    value_false = jsgraph()->UndefinedConstant();

    if (!IsFixedTypedArrayElementsKind(elements_kind)) {
      // Mark the {iterator} as exhausted by setting the [[NextIndex]] to a
      // value that will never pass the length check again (aka the maximum
      // value possible for the specific iterated object). Note that this is
      // different from what the specification says, which is changing the
      // [[IteratedObject]] field to undefined, but that makes it difficult
      // to eliminate the map checks and "length" accesses in for..of loops.
      //
      // This is not necessary for JSTypedArray's, since the length of those
      // cannot change later and so if we were ever out of bounds for them
      // we will stay out-of-bounds forever.
      Node* end_index = jsgraph()->Constant(index_access.type.Max());
      efalse = graph()->NewNode(simplified()->StoreField(index_access),
                                iterator, end_index, efalse, if_false);
    }
  }

  control = graph()->NewNode(common()->Merge(2), if_true, if_false);
  effect = graph()->NewNode(common()->EffectPhi(2), etrue, efalse, control);
  Node* value =
      graph()->NewNode(common()->Phi(MachineRepresentation::kTagged, 2),
                       value_true, value_false, control);
  Node* done =
      graph()->NewNode(common()->Phi(MachineRepresentation::kTagged, 2),
                       done_true, done_false, control);

  // Create IteratorResult object.
  value = effect = graph()->NewNode(javascript()->CreateIterResultObject(),
                                    value, done, context, effect);
  ReplaceWithValue(node, value, effect, control);
  return Replace(value);
}
