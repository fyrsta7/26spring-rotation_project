bool LinearScanAllocator::TryAllocateFreeReg(
    LiveRange* current, const Vector<LifetimePosition>& free_until_pos) {
  int num_regs = 0;  // used only for the call to GetFPRegisterSet.
  int num_codes = num_allocatable_registers();
  const int* codes = allocatable_register_codes();
  MachineRepresentation rep = current->representation();
  if (!kSimpleFPAliasing && (rep == MachineRepresentation::kFloat32 ||
                             rep == MachineRepresentation::kSimd128)) {
    GetFPRegisterSet(rep, &num_regs, &num_codes, &codes);
  }

  DCHECK_GE(free_until_pos.length(), num_codes);

  // Find the register which stays free for the longest time. Check for
  // the hinted register first, as we might want to use that one. Only
  // count full instructions for free ranges, as an instruction's internal
  // positions do not help but might shadow a hinted register. This is
  // typically the case for function calls, where all registered are
  // cloberred after the call except for the argument registers, which are
  // set before the call. Hence, the argument registers always get ignored,
  // as their available time is shorter.
  int reg;
  if (current->FirstHintPosition(&reg) == nullptr) {
    reg = codes[0];
  }
  for (int i = 0; i < num_codes; ++i) {
    int code = codes[i];
    if (free_until_pos[code].ToInstructionIndex() >
        free_until_pos[reg].ToInstructionIndex()) {
      reg = code;
    }
  }

  LifetimePosition pos = free_until_pos[reg];

  if (pos <= current->Start()) {
    // All registers are blocked.
    return false;
  }

  if (pos < current->End()) {
    // Register reg is available at the range start but becomes blocked before
    // the range end. Split current at position where it becomes blocked.
    LiveRange* tail = SplitRangeAt(current, pos);
    AddToUnhandled(tail);

    // Try to allocate preferred register once more.
    if (TryAllocatePreferredReg(current, free_until_pos)) return true;
  }

  // Register reg is available at the range start and is free until the range
  // end.
  DCHECK(pos >= current->End());
  TRACE("Assigning free reg %s to live range %d:%d\n", RegisterName(reg),
        current->TopLevel()->vreg(), current->relative_id());
  SetLiveRangeAssignedRegister(current, reg);

  return true;
}
