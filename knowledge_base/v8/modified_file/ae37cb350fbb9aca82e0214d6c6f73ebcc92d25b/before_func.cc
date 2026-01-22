void LCodeGen::DoDeferredAllocate(LAllocate* instr) {
  Register size = ToRegister(instr->size());
  Register result = ToRegister(instr->result());

  // TODO(3095996): Get rid of this. For now, we need to make the
  // result register contain a valid pointer because it is already
  // contained in the register pointer map.
  __ mov(result, zero_reg);

  PushSafepointRegistersScope scope(this, Safepoint::kWithRegisters);
  __ SmiTag(size, size);
  __ push(size);
  CallRuntimeFromDeferred(Runtime::kAllocateInNewSpace, 1, instr);
  __ StoreToSafepointRegisterSlot(v0, result);
}
