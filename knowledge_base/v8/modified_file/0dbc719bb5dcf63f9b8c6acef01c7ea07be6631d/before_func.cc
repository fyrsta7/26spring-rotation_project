Node* CodeStubAssembler::LoadDoubleWithHoleCheck(Node* base, Node* offset,
                                                 Label* if_hole,
                                                 MachineType machine_type) {
  if (if_hole) {
    Node* element_upper =
        Load(MachineType::Uint32(), base,
             IntPtrAdd(offset, IntPtrConstant(kIeeeDoubleExponentWordOffset)));
    GotoIf(Word32Equal(element_upper, Int32Constant(kHoleNanUpper32)), if_hole);
  }
  if (machine_type.IsNone()) {
    // This means the actual value is not needed.
    return nullptr;
  }
  return Load(machine_type, base, offset);
}
