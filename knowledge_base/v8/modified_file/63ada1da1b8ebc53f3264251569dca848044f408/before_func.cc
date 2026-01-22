void LCodeGen::DoLoadKeyedFixedDoubleArray(LLoadKeyed* instr) {
  Register elements = ToRegister(instr->elements());
  bool key_is_constant = instr->key()->IsConstantOperand();
  Register key = no_reg;
  DwVfpRegister result = ToDoubleRegister(instr->result());
  Register scratch = scratch0();

  int element_size_shift = ElementsKindToShiftSize(FAST_DOUBLE_ELEMENTS);
  int shift_size = (instr->hydrogen()->key()->representation().IsTagged())
      ? (element_size_shift - kSmiTagSize) : element_size_shift;
  int constant_key = 0;
  if (key_is_constant) {
    constant_key = ToInteger32(LConstantOperand::cast(instr->key()));
    if (constant_key & 0xF0000000) {
      Abort("array index constant value too big.");
    }
  } else {
    key = ToRegister(instr->key());
  }

  Operand operand = key_is_constant
      ? Operand(((constant_key + instr->additional_index()) <<
                 element_size_shift) +
                FixedDoubleArray::kHeaderSize - kHeapObjectTag)
      : Operand(key, LSL, shift_size);
  __ add(elements, elements, operand);
  if (!key_is_constant) {
    __ add(elements, elements,
           Operand((FixedDoubleArray::kHeaderSize - kHeapObjectTag) +
                   (instr->additional_index() << element_size_shift)));
  }

  if (instr->hydrogen()->RequiresHoleCheck()) {
    __ ldr(scratch, MemOperand(elements, sizeof(kHoleNanLower32)));
    __ cmp(scratch, Operand(kHoleNanUpper32));
    DeoptimizeIf(eq, instr->environment());
  }

  __ vldr(result, elements, 0);
}
