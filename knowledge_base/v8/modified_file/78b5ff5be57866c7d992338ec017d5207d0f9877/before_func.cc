void SinglePassRegisterAllocator::CheckConsistency() {
#ifdef DEBUG
  for (int virtual_register = 0;
       virtual_register < data_->code()->VirtualRegisterCount();
       virtual_register++) {
    RegisterIndex reg = RegisterForVirtualRegister(virtual_register);
    if (!reg.is_valid()) continue;
    CHECK_NOT_NULL(register_state_);
    // The register must be set to allocated.
    CHECK(register_state_->IsAllocated(reg));
    // reg <-> vreg linking is consistent.
    CHECK_EQ(virtual_register, VirtualRegisterForRegister(reg));
  }

  RegisterBitVector used_registers;
  for (RegisterIndex reg : *register_state_) {
    if (!register_state_->IsAllocated(reg)) continue;
    int virtual_register = VirtualRegisterForRegister(reg);
    // reg <-> vreg linking is consistent.
    CHECK_EQ(reg, RegisterForVirtualRegister(virtual_register));
    MachineRepresentation rep = VirtualRegisterDataFor(virtual_register).rep();
    // Allocated registers do not overlap.
    CHECK(!used_registers.Contains(reg, rep));
    used_registers.Add(reg, rep);
  }
  // The {allocated_registers_bits_} bitvector is accurate.
  CHECK_EQ(used_registers, allocated_registers_bits_);
#endif
}
