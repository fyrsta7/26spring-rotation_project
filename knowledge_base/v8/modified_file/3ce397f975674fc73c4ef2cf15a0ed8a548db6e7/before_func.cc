LInstruction* LChunkBuilder::DoBranch(HBranch* instr) {
  HValue* v = instr->value();
  if (v->EmitAtUses()) {
    ASSERT(v->IsConstant());
    ASSERT(!v->representation().IsDouble());
    HBasicBlock* successor = HConstant::cast(v)->ToBoolean()
        ? instr->FirstSuccessor()
        : instr->SecondSuccessor();
    return new LGoto(successor->block_id());
  }
  ToBooleanStub::Types expected = instr->expected_input_types();
  bool needs_temp = expected.NeedsMap() || expected.IsEmpty();
  LOperand* temp = needs_temp ? TempRegister() : NULL;
  return AssignEnvironment(new LBranch(UseRegister(v), temp));
}
