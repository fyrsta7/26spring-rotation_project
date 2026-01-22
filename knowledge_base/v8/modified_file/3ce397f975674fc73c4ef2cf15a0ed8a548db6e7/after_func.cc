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
  // We need a temporary register when we have to access the map *or* we have
  // no type info yet, in which case we handle all cases (including the ones
  // involving maps).
  bool needs_temp = expected.NeedsMap() || expected.IsEmpty();
  LOperand* temp = needs_temp ? TempRegister() : NULL;
  LInstruction* branch = new LBranch(UseRegister(v), temp);
  // When we handle all cases, we never deopt, so we don't need to assign the
  // environment then.
  return expected.IsAll() ? branch : AssignEnvironment(branch);
}
