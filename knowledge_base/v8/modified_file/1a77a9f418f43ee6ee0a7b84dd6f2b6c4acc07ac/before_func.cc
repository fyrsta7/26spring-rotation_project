InductionVariable* LoopVariableOptimizer::TryGetInductionVariable(Node* phi) {
  DCHECK_EQ(2, phi->op()->ValueInputCount());
  Node* loop = NodeProperties::GetControlInput(phi);
  DCHECK_EQ(IrOpcode::kLoop, loop->opcode());
  Node* initial = phi->InputAt(0);
  Node* arith = phi->InputAt(1);
  InductionVariable::ArithmeticType arithmeticType;
  if (arith->opcode() == IrOpcode::kJSAdd ||
      arith->opcode() == IrOpcode::kSpeculativeNumberAdd ||
      arith->opcode() == IrOpcode::kSpeculativeSafeIntegerAdd) {
    arithmeticType = InductionVariable::ArithmeticType::kAddition;
  } else if (arith->opcode() == IrOpcode::kJSSubtract ||
             arith->opcode() == IrOpcode::kSpeculativeNumberSubtract ||
             arith->opcode() == IrOpcode::kSpeculativeSafeIntegerSubtract) {
    arithmeticType = InductionVariable::ArithmeticType::kSubtraction;
  } else {
    return nullptr;
  }

  // TODO(jarin) Support both sides.
  if (arith->InputAt(0) != phi) return nullptr;

  Node* effect_phi = nullptr;
  for (Node* use : loop->uses()) {
    if (use->opcode() == IrOpcode::kEffectPhi) {
      DCHECK_NULL(effect_phi);
      effect_phi = use;
    }
  }
  if (!effect_phi) return nullptr;

  Node* incr = arith->InputAt(1);
  return new (zone()) InductionVariable(phi, effect_phi, arith, incr, initial,
                                        zone(), arithmeticType);
}
