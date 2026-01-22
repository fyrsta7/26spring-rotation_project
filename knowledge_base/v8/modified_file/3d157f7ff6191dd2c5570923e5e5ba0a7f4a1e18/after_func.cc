InductionVariable* LoopVariableOptimizer::TryGetInductionVariable(Node* phi) {
  DCHECK_EQ(2, phi->op()->ValueInputCount());
  DCHECK_EQ(IrOpcode::kLoop, NodeProperties::GetControlInput(phi)->opcode());
  Node* initial = phi->InputAt(0);
  Node* arith = phi->InputAt(1);
  InductionVariable::ArithmeticType arithmeticType;
  if (arith->opcode() == IrOpcode::kJSAdd ||
      arith->opcode() == IrOpcode::kSpeculativeNumberAdd) {
    arithmeticType = InductionVariable::ArithmeticType::kAddition;
  } else if (arith->opcode() == IrOpcode::kJSSubtract ||
             arith->opcode() == IrOpcode::kSpeculativeNumberSubtract) {
    arithmeticType = InductionVariable::ArithmeticType::kSubtraction;
  } else {
    return nullptr;
  }

  // TODO(jarin) Support both sides.
  if (arith->InputAt(0) != phi) {
    if (arith->InputAt(0)->opcode() != IrOpcode::kJSToNumber ||
        arith->InputAt(0)->InputAt(0) != phi) {
      return nullptr;
    }
  }
  Node* incr = arith->InputAt(1);
  return new (zone())
      InductionVariable(phi, arith, incr, initial, zone(), arithmeticType);
}
