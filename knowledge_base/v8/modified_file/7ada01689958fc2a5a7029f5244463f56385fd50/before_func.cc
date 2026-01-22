Reduction SimplifiedOperatorReducer::Reduce(Node* node) {
  switch (node->opcode()) {
    case IrOpcode::kBooleanNot: {
      HeapObjectMatcher m(node->InputAt(0));
      if (m.Is(factory()->true_value())) return ReplaceBoolean(false);
      if (m.Is(factory()->false_value())) return ReplaceBoolean(true);
      if (m.IsBooleanNot()) return Replace(m.InputAt(0));
      break;
    }
    case IrOpcode::kChangeBitToTagged: {
      Int32Matcher m(node->InputAt(0));
      if (m.Is(0)) return Replace(jsgraph()->FalseConstant());
      if (m.Is(1)) return Replace(jsgraph()->TrueConstant());
      if (m.IsChangeTaggedToBit()) return Replace(m.InputAt(0));
      break;
    }
    case IrOpcode::kChangeTaggedToBit: {
      HeapObjectMatcher m(node->InputAt(0));
      if (m.HasValue()) return ReplaceInt32(m.Value()->BooleanValue());
      if (m.IsChangeBitToTagged()) return Replace(m.InputAt(0));
      break;
    }
    case IrOpcode::kChangeFloat64ToTagged: {
      Float64Matcher m(node->InputAt(0));
      if (m.HasValue()) return ReplaceNumber(m.Value());
      if (m.IsChangeTaggedToFloat64()) return Replace(m.node()->InputAt(0));
      break;
    }
    case IrOpcode::kChangeInt31ToTaggedSigned:
    case IrOpcode::kChangeInt32ToTagged: {
      Int32Matcher m(node->InputAt(0));
      if (m.HasValue()) return ReplaceNumber(m.Value());
      if (m.IsChangeTaggedToInt32() || m.IsChangeTaggedSignedToInt32()) {
        return Replace(m.InputAt(0));
      }
      break;
    }
    case IrOpcode::kChangeTaggedToFloat64:
    case IrOpcode::kTruncateTaggedToFloat64: {
      NumberMatcher m(node->InputAt(0));
      if (m.HasValue()) return ReplaceFloat64(m.Value());
      if (m.IsChangeFloat64ToTagged()) return Replace(m.node()->InputAt(0));
      if (m.IsChangeInt31ToTaggedSigned() || m.IsChangeInt32ToTagged()) {
        return Change(node, machine()->ChangeInt32ToFloat64(), m.InputAt(0));
      }
      if (m.IsChangeUint32ToTagged()) {
        return Change(node, machine()->ChangeUint32ToFloat64(), m.InputAt(0));
      }
      break;
    }
    case IrOpcode::kChangeTaggedToInt32: {
      NumberMatcher m(node->InputAt(0));
      if (m.HasValue()) return ReplaceInt32(DoubleToInt32(m.Value()));
      if (m.IsChangeFloat64ToTagged()) {
        return Change(node, machine()->ChangeFloat64ToInt32(), m.InputAt(0));
      }
      if (m.IsChangeInt31ToTaggedSigned() || m.IsChangeInt32ToTagged()) {
        return Replace(m.InputAt(0));
      }
      break;
    }
    case IrOpcode::kChangeTaggedToUint32: {
      NumberMatcher m(node->InputAt(0));
      if (m.HasValue()) return ReplaceUint32(DoubleToUint32(m.Value()));
      if (m.IsChangeFloat64ToTagged()) {
        return Change(node, machine()->ChangeFloat64ToUint32(), m.InputAt(0));
      }
      if (m.IsChangeUint32ToTagged()) return Replace(m.InputAt(0));
      break;
    }
    case IrOpcode::kChangeUint32ToTagged: {
      Uint32Matcher m(node->InputAt(0));
      if (m.HasValue()) return ReplaceNumber(FastUI2D(m.Value()));
      break;
    }
    case IrOpcode::kTruncateTaggedToWord32: {
      NumberMatcher m(node->InputAt(0));
      if (m.HasValue()) return ReplaceInt32(DoubleToInt32(m.Value()));
      if (m.IsChangeInt31ToTaggedSigned() || m.IsChangeInt32ToTagged() ||
          m.IsChangeUint32ToTagged()) {
        return Replace(m.InputAt(0));
      }
      if (m.IsChangeFloat64ToTagged()) {
        return Change(node, machine()->TruncateFloat64ToWord32(), m.InputAt(0));
      }
      break;
    }
    case IrOpcode::kCheckIf: {
      HeapObjectMatcher m(node->InputAt(0));
      if (m.Is(factory()->true_value())) {
        Node* const effect = NodeProperties::GetEffectInput(node);
        return Replace(effect);
      }
      break;
    }
    case IrOpcode::kCheckTaggedPointer: {
      Node* const input = node->InputAt(0);
      if (DecideObjectIsSmi(input) == Decision::kFalse) {
        ReplaceWithValue(node, input);
        return Replace(input);
      }
      break;
    }
    case IrOpcode::kCheckTaggedSigned: {
      Node* const input = node->InputAt(0);
      if (DecideObjectIsSmi(input) == Decision::kTrue) {
        ReplaceWithValue(node, input);
        return Replace(input);
      }
      break;
    }
    case IrOpcode::kObjectIsSmi: {
      Node* const input = node->InputAt(0);
      switch (DecideObjectIsSmi(input)) {
        case Decision::kTrue:
          return ReplaceBoolean(true);
        case Decision::kFalse:
          return ReplaceBoolean(false);
        case Decision::kUnknown:
          break;
      }
      break;
    }
    case IrOpcode::kNumberAbs: {
      NumberMatcher m(node->InputAt(0));
      if (m.HasValue()) return ReplaceNumber(std::fabs(m.Value()));
      break;
    }
    case IrOpcode::kReferenceEqual: {
      HeapObjectBinopMatcher m(node);
      if (m.left().node() == m.right().node()) return ReplaceBoolean(true);
      break;
    }
    default:
      break;
  }
  return NoChange();
}
