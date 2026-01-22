Reduction JSBuiltinReducer::ReduceArrayPop(Node* node) {
  Handle<Map> receiver_map;
  Node* receiver = NodeProperties::GetValueInput(node, 1);
  Node* effect = NodeProperties::GetEffectInput(node);
  Node* control = NodeProperties::GetControlInput(node);
  // TODO(turbofan): Extend this to also handle fast (holey) double elements
  // once we got the hole NaN mess sorted out in TurboFan/V8.
  if (GetMapWitness(node).ToHandle(&receiver_map) &&
      CanInlineArrayResizeOperation(receiver_map) &&
      IsFastSmiOrObjectElementsKind(receiver_map->elements_kind())) {
    // Install code dependencies on the {receiver} prototype maps and the
    // global array protector cell.
    dependencies()->AssumePropertyCell(factory()->array_protector());
    dependencies()->AssumePrototypeMapsStable(receiver_map);

    // Load the "length" property of the {receiver}.
    Node* length = effect = graph()->NewNode(
        simplified()->LoadField(
            AccessBuilder::ForJSArrayLength(receiver_map->elements_kind())),
        receiver, effect, control);

    // Check if the {receiver} has any elements.
    Node* check = graph()->NewNode(simplified()->NumberEqual(), length,
                                   jsgraph()->ZeroConstant());
    Node* branch =
        graph()->NewNode(common()->Branch(BranchHint::kFalse), check, control);

    Node* if_true = graph()->NewNode(common()->IfTrue(), branch);
    Node* etrue = effect;
    Node* vtrue = jsgraph()->UndefinedConstant();

    Node* if_false = graph()->NewNode(common()->IfFalse(), branch);
    Node* efalse = effect;
    Node* vfalse;
    {
      // Load the elements backing store from the {receiver}.
      Node* elements = efalse = graph()->NewNode(
          simplified()->LoadField(AccessBuilder::ForJSObjectElements()),
          receiver, efalse, if_false);

      // Ensure that we aren't popping from a copy-on-write backing store.
      elements = efalse =
          graph()->NewNode(simplified()->EnsureWritableFastElements(), receiver,
                           elements, efalse, if_false);

      // Compute the new {length}.
      length = graph()->NewNode(simplified()->NumberSubtract(), length,
                                jsgraph()->OneConstant());

      // Store the new {length} to the {receiver}.
      efalse = graph()->NewNode(
          simplified()->StoreField(
              AccessBuilder::ForJSArrayLength(receiver_map->elements_kind())),
          receiver, length, efalse, if_false);

      // Load the last entry from the {elements}.
      vfalse = efalse = graph()->NewNode(
          simplified()->LoadElement(AccessBuilder::ForFixedArrayElement(
              receiver_map->elements_kind())),
          elements, length, efalse, if_false);

      // Store a hole to the element we just removed from the {receiver}.
      efalse = graph()->NewNode(
          simplified()->StoreElement(AccessBuilder::ForFixedArrayElement(
              GetHoleyElementsKind(receiver_map->elements_kind()))),
          elements, length, jsgraph()->TheHoleConstant(), efalse, if_false);
    }

    control = graph()->NewNode(common()->Merge(2), if_true, if_false);
    effect = graph()->NewNode(common()->EffectPhi(2), etrue, efalse, control);
    Node* value =
        graph()->NewNode(common()->Phi(MachineRepresentation::kTagged, 2),
                         vtrue, vfalse, control);

    // Convert the hole to undefined. Do this last, so that we can optimize
    // conversion operator via some smart strength reduction in many cases.
    if (IsFastHoleyElementsKind(receiver_map->elements_kind())) {
      value =
          graph()->NewNode(simplified()->ConvertTaggedHoleToUndefined(), value);
    }

    ReplaceWithValue(node, value, effect, control);
    return Replace(value);
  }
  return NoChange();
}
