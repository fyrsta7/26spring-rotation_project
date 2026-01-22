LoadElimination::AbstractState const* LoadElimination::ComputeLoopState(
    Node* node, AbstractState const* state) const {
  Node* const control = NodeProperties::GetControlInput(node);
  ZoneQueue<Node*> queue(zone());
  ZoneSet<Node*> visited(zone());
  visited.insert(node);
  for (int i = 1; i < control->InputCount(); ++i) {
    queue.push(node->InputAt(i));
  }
  while (!queue.empty()) {
    Node* const current = queue.front();
    queue.pop();
    if (visited.find(current) == visited.end()) {
      visited.insert(current);
      if (!current->op()->HasProperty(Operator::kNoWrite)) {
        switch (current->opcode()) {
          case IrOpcode::kEnsureWritableFastElements: {
            Node* const object = NodeProperties::GetValueInput(current, 0);
            state = state->KillField(
                object, FieldIndexOf(JSObject::kElementsOffset), zone());
            break;
          }
          case IrOpcode::kMaybeGrowFastElements: {
            GrowFastElementsFlags flags =
                GrowFastElementsFlagsOf(current->op());
            Node* const object = NodeProperties::GetValueInput(current, 0);
            state = state->KillField(
                object, FieldIndexOf(JSObject::kElementsOffset), zone());
            if (flags & GrowFastElementsFlag::kArrayObject) {
              state = state->KillField(
                  object, FieldIndexOf(JSArray::kLengthOffset), zone());
            }
            break;
          }
          case IrOpcode::kTransitionElementsKind: {
            Node* const object = NodeProperties::GetValueInput(current, 0);
            Node* const target_map = NodeProperties::GetValueInput(current, 2);
            Node* const object_map = state->LookupField(
                object, FieldIndexOf(HeapObject::kMapOffset));
            if (target_map != object_map) {
              state = state->KillField(
                  object, FieldIndexOf(HeapObject::kMapOffset), zone());
              state = state->KillField(
                  object, FieldIndexOf(JSObject::kElementsOffset), zone());
            }
            break;
          }
          case IrOpcode::kStoreField: {
            FieldAccess const& access = FieldAccessOf(current->op());
            Node* const object = NodeProperties::GetValueInput(current, 0);
            int field_index = FieldIndexOf(access);
            if (field_index < 0) {
              state = state->KillFields(object, zone());
            } else {
              state = state->KillField(object, field_index, zone());
            }
            break;
          }
          case IrOpcode::kStoreElement: {
            Node* const object = NodeProperties::GetValueInput(current, 0);
            Node* const index = NodeProperties::GetValueInput(current, 1);
            state = state->KillElement(object, index, zone());
            break;
          }
          case IrOpcode::kStoreBuffer:
          case IrOpcode::kStoreTypedElement: {
            // Doesn't affect anything we track with the state currently.
            break;
          }
          default:
            return empty_state();
        }
      }
      for (int i = 0; i < current->op()->EffectInputCount(); ++i) {
        queue.push(NodeProperties::GetEffectInput(current, i));
      }
    }
  }
  return state;
}
