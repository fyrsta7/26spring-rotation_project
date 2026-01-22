void EmitWordCompareZero(InstructionSelector* selector, Node* value,
                         FlagsContinuation* cont) {
  Mips64OperandGenerator g(selector);
  InstructionCode opcode = cont->Encode(kMips64Cmp);
  InstructionOperand const value_operand = g.UseRegister(value);
  if (cont->IsBranch()) {
    selector->Emit(opcode, g.NoOutput(), value_operand, g.TempImmediate(0),
                   g.Label(cont->true_block()), g.Label(cont->false_block()));
  } else {
    selector->Emit(opcode, g.DefineAsRegister(cont->result()), value_operand,
                   g.TempImmediate(0));
  }
}
