void LCodeGen::DoCheckMaps(LCheckMaps* instr) {
  Register scratch = scratch0();
  LOperand* input = instr->InputAt(0);
  ASSERT(input->IsRegister());
  Register reg = ToRegister(input);
  Label success;
  SmallMapList* map_set = instr->hydrogen()->map_set();
  for (int i = 0; i < map_set->length() - 1; i++) {
    Handle<Map> map = map_set->at(i);
    __ CompareMapAndBranch(
        reg, scratch, map, &success, eq, &success, REQUIRE_EXACT_MAP);
  }
  Handle<Map> map = map_set->last();
  DoCheckMapCommon(reg, scratch, map, REQUIRE_EXACT_MAP, instr->environment());
  __ bind(&success);
}
