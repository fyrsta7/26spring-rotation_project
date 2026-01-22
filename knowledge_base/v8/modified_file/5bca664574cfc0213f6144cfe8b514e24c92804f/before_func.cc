void LCodeGen::DoCheckMaps(LCheckMaps* instr) {
  Register scratch = scratch0();
  LOperand* input = instr->InputAt(0);
  ASSERT(input->IsRegister());
  Register reg = ToRegister(input);
  Handle<Map> map = instr->hydrogen()->map();
  DoCheckMapCommon(reg, scratch, map, instr->hydrogen()->mode(),
                   instr->environment());
}
