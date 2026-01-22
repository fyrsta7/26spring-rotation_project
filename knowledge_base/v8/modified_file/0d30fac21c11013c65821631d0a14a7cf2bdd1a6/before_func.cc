bool Bytecodes::IsStarLookahead(Bytecode bytecode, OperandScale operand_scale) {
  if (operand_scale == OperandScale::kSingle) {
    switch (bytecode) {
      case Bytecode::kLdaZero:
      case Bytecode::kLdaSmi:
      case Bytecode::kLdaNull:
      case Bytecode::kLdaTheHole:
      case Bytecode::kLdaConstant:
      case Bytecode::kLdaUndefined:
      case Bytecode::kLdaGlobal:
      case Bytecode::kLdaNamedProperty:
      case Bytecode::kLdaKeyedProperty:
      case Bytecode::kLdaContextSlot:
      case Bytecode::kLdaCurrentContextSlot:
      case Bytecode::kAdd:
      case Bytecode::kSub:
      case Bytecode::kMul:
      case Bytecode::kAddSmi:
      case Bytecode::kSubSmi:
      case Bytecode::kInc:
      case Bytecode::kDec:
      case Bytecode::kTypeOf:
      case Bytecode::kCallAnyReceiver:
      case Bytecode::kCallNoFeedback:
      case Bytecode::kCallProperty:
      case Bytecode::kCallProperty0:
      case Bytecode::kCallProperty1:
      case Bytecode::kCallProperty2:
      case Bytecode::kCallUndefinedReceiver:
      case Bytecode::kCallUndefinedReceiver0:
      case Bytecode::kCallUndefinedReceiver1:
      case Bytecode::kCallUndefinedReceiver2:
      case Bytecode::kConstruct:
      case Bytecode::kConstructWithSpread:
        return true;
      default:
        return false;
    }
  }
  return false;
}
