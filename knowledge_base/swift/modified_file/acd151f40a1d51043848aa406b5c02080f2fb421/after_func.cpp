static unsigned instructionInlineCost(SILInstruction &I) {
  switch (I.getKind()) {
    case ValueKind::FunctionRefInst:
    case ValueKind::BuiltinFunctionRefInst:
    case ValueKind::GlobalAddrInst:
    case ValueKind::SILGlobalAddrInst:
    case ValueKind::IntegerLiteralInst:
    case ValueKind::FloatLiteralInst:
    case ValueKind::DebugValueInst:
    case ValueKind::DebugValueAddrInst:
      return 0;
    case ValueKind::TupleElementAddrInst:
    case ValueKind::StructElementAddrInst: {
      // A gep whose operand is a gep with no other users will get folded by
      // LLVM into one gep implying the second should be free.
      SILValue Op = I.getOperand(0);
      if ((Op->getKind() == ValueKind::TupleElementAddrInst ||
           Op->getKind() == ValueKind::StructElementAddrInst) &&
          Op->hasOneUse())
        return 0;
    }
    default:
      return 1;
  }
}
