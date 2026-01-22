void LCodeGen::DoCheckMaps(LCheckMaps* instr) {
  class DeferredCheckMaps: public LDeferredCode {
   public:
    DeferredCheckMaps(LCodeGen* codegen, LCheckMaps* instr, Register object)
        : LDeferredCode(codegen), instr_(instr), object_(object) {
      SetExit(check_maps());
    }
    virtual void Generate() {
      codegen()->DoDeferredInstanceMigration(instr_, object_);
    }
    Label* check_maps() { return &check_maps_; }
    virtual LInstruction* instr() { return instr_; }
   private:
    LCheckMaps* instr_;
    Label check_maps_;
    Register object_;
  };

  if (instr->hydrogen()->CanOmitMapChecks()) return;
  Register map_reg = scratch0();
  LOperand* input = instr->value();
  ASSERT(input->IsRegister());
  Register reg = ToRegister(input);
  SmallMapList* map_set = instr->hydrogen()->map_set();
  __ lw(map_reg, FieldMemOperand(reg, HeapObject::kMapOffset));

  DeferredCheckMaps* deferred = NULL;
  if (instr->hydrogen()->has_migration_target()) {
    deferred = new(zone()) DeferredCheckMaps(this, instr, reg);
    __ bind(deferred->check_maps());
  }

  Label success;
  for (int i = 0; i < map_set->length() - 1; i++) {
    Handle<Map> map = map_set->at(i);
    __ CompareMapAndBranch(map_reg, map, &success, eq, &success);
  }
  Handle<Map> map = map_set->last();
  __ CompareMapAndBranch(map_reg, map, &success, eq, &success);
  if (instr->hydrogen()->has_migration_target()) {
    __ Branch(deferred->entry());
  } else {
    DeoptimizeIf(al, instr->environment());
  }

  __ bind(&success);
}
