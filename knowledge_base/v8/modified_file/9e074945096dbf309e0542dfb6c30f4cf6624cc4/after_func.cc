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

  // Find the register which stays free for the longest time.
  int reg = codes[0];
  for (int i = 1; i < num_codes; ++i) {
    int code = codes[i];
    if (free_until_pos[code] > free_until_pos[reg]) {
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
    AddToUnhandledSorted(tail);

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
