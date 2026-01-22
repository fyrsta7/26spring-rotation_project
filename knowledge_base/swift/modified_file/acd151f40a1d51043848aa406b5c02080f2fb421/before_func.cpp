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
    default:
      return 1;
  }
}
