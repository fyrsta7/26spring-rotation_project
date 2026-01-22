Reduction JSCallReducer::ReduceFunctionPrototypeBind(Node* node) {
  DCHECK_EQ(IrOpcode::kJSCall, node->opcode());
  // Value inputs to the {node} are as follows:
  //
  //  - target, which is Function.prototype.bind JSFunction
  //  - receiver, which is the [[BoundTargetFunction]]
  //  - bound_this (optional), which is the [[BoundThis]]
  //  - and all the remaining value inouts are [[BoundArguments]]
  Node* receiver = NodeProperties::GetValueInput(node, 1);
  Node* bound_this = (node->op()->ValueInputCount() < 3)
                         ? jsgraph()->UndefinedConstant()
                         : NodeProperties::GetValueInput(node, 2);
  Node* context = NodeProperties::GetContextInput(node);
  Node* effect = NodeProperties::GetEffectInput(node);
  Node* control = NodeProperties::GetControlInput(node);

  // Ensure that the {receiver} is known to be a JSBoundFunction or
  // a JSFunction with the same [[Prototype]], and all maps we've
  // seen for the {receiver} so far indicate that {receiver} is
  // definitely a constructor or not a constructor.
  ZoneHandleSet<Map> receiver_maps;
  NodeProperties::InferReceiverMapsResult result =
      NodeProperties::InferReceiverMaps(broker(), receiver, effect,
                                        &receiver_maps);
  if (result == NodeProperties::kNoReceiverMaps) return NoChange();
  DCHECK_NE(0, receiver_maps.size());
  bool const is_constructor = receiver_maps[0]->is_constructor();
  Handle<Object> const prototype(receiver_maps[0]->prototype(), isolate());
  for (Handle<Map> const receiver_map : receiver_maps) {
    // Check for consistency among the {receiver_maps}.
    STATIC_ASSERT(LAST_TYPE == LAST_FUNCTION_TYPE);
    if (receiver_map->prototype() != *prototype) return NoChange();
    if (receiver_map->is_constructor() != is_constructor) return NoChange();
    if (receiver_map->instance_type() < FIRST_FUNCTION_TYPE) return NoChange();

    // Disallow binding of slow-mode functions. We need to figure out
    // whether the length and name property are in the original state.
    if (receiver_map->is_dictionary_map()) return NoChange();

    // Check whether the length and name properties are still present
    // as AccessorInfo objects. In that case, their values can be
    // recomputed even if the actual value of the object changes.
    // This mirrors the checks done in builtins-function-gen.cc at
    // runtime otherwise.
    Handle<DescriptorArray> descriptors(receiver_map->instance_descriptors(),
                                        isolate());
    if (descriptors->number_of_descriptors() < 2) return NoChange();
    if (descriptors->GetKey(JSFunction::kLengthDescriptorIndex) !=
        ReadOnlyRoots(isolate()).length_string()) {
      return NoChange();
    }
    if (!descriptors->GetStrongValue(JSFunction::kLengthDescriptorIndex)
             ->IsAccessorInfo()) {
      return NoChange();
    }
    if (descriptors->GetKey(JSFunction::kNameDescriptorIndex) !=
        ReadOnlyRoots(isolate()).name_string()) {
      return NoChange();
    }
    if (!descriptors->GetStrongValue(JSFunction::kNameDescriptorIndex)
             ->IsAccessorInfo()) {
      return NoChange();
    }
  }

  // Setup the map for the resulting JSBoundFunction with the
  // correct instance {prototype}.
  Handle<Map> map(
      is_constructor
          ? native_context()->bound_function_with_constructor_map()
          : native_context()->bound_function_without_constructor_map(),
      isolate());
  if (map->prototype() != *prototype) {
    map = Map::TransitionToPrototype(isolate(), map, prototype);
  }

  // Make sure we can rely on the {receiver_maps}.
  if (result == NodeProperties::kUnreliableReceiverMaps) {
    effect = graph()->NewNode(
        simplified()->CheckMaps(CheckMapsFlag::kNone, receiver_maps), receiver,
        effect, control);
  }

  // Replace the {node} with a JSCreateBoundFunction.
  int const arity = std::max(0, node->op()->ValueInputCount() - 3);
  int const input_count = 2 + arity + 3;
  Node** inputs = graph()->zone()->NewArray<Node*>(input_count);
  inputs[0] = receiver;
  inputs[1] = bound_this;
  for (int i = 0; i < arity; ++i) {
    inputs[2 + i] = NodeProperties::GetValueInput(node, 3 + i);
  }
  inputs[2 + arity + 0] = context;
  inputs[2 + arity + 1] = effect;
  inputs[2 + arity + 2] = control;
  Node* value = effect = graph()->NewNode(
      javascript()->CreateBoundFunction(arity, map), input_count, inputs);
  ReplaceWithValue(node, value, effect, control);
  return Replace(value);
}
