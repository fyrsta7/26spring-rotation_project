Reduction RedundancyElimination::Reduce(Node* node) {
  if (node_checks_.Get(node)) return NoChange();
  switch (node->opcode()) {
    case IrOpcode::kCheckBigInt:
    case IrOpcode::kCheckBounds:
    case IrOpcode::kCheckClosure:
    case IrOpcode::kCheckEqualsInternalizedString:
    case IrOpcode::kCheckEqualsSymbol:
    case IrOpcode::kCheckFloat64Hole:
    case IrOpcode::kCheckHeapObject:
    case IrOpcode::kCheckIf:
    case IrOpcode::kCheckInternalizedString:
    case IrOpcode::kCheckNotTaggedHole:
    case IrOpcode::kCheckNumber:
    case IrOpcode::kCheckReceiver:
    case IrOpcode::kCheckReceiverOrNullOrUndefined:
    case IrOpcode::kCheckSmi:
    case IrOpcode::kCheckString:
    case IrOpcode::kCheckSymbol:
    // These are not really check nodes, but behave the same in that they can be
    // folded together if repeated with identical inputs.
    case IrOpcode::kBigIntAdd:
    case IrOpcode::kBigIntSubtract:
    case IrOpcode::kStringCharCodeAt:
    case IrOpcode::kStringCodePointAt:
    case IrOpcode::kStringFromCodePointAt:
    case IrOpcode::kStringSubstring:
#define SIMPLIFIED_CHECKED_OP(Opcode) case IrOpcode::k##Opcode:
      SIMPLIFIED_CHECKED_OP_LIST(SIMPLIFIED_CHECKED_OP)
#undef SIMPLIFIED_CHECKED_OP
      return ReduceCheckNode(node);
    case IrOpcode::kSpeculativeNumberEqual:
    case IrOpcode::kSpeculativeNumberLessThan:
    case IrOpcode::kSpeculativeNumberLessThanOrEqual:
      return ReduceSpeculativeNumberComparison(node);
    case IrOpcode::kSpeculativeNumberAdd:
    case IrOpcode::kSpeculativeNumberSubtract:
    case IrOpcode::kSpeculativeSafeIntegerAdd:
    case IrOpcode::kSpeculativeSafeIntegerSubtract:
    case IrOpcode::kSpeculativeToNumber:
      return ReduceSpeculativeNumberOperation(node);
    case IrOpcode::kEffectPhi:
      return ReduceEffectPhi(node);
    case IrOpcode::kDead:
      break;
    case IrOpcode::kStart:
      return ReduceStart(node);
    default:
      return ReduceOtherNode(node);
  }
  return NoChange();
}
