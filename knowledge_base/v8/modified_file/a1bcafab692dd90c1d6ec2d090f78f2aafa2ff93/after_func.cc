void MacroAssembler::RecordWriteHelper(Register object,
                                       Register addr,
                                       Register scratch) {
  if (emit_debug_code()) {
    // Check that the object is not in new space.
    Label not_in_new_space;
    InNewSpace(object, scratch, not_equal, &not_in_new_space);
    Abort("new-space object passed to RecordWriteHelper");
    bind(&not_in_new_space);
  }

  // Compute the page start address from the heap object pointer, and reuse
  // the 'object' register for it.
  and_(object, ~Page::kPageAlignmentMask);

  // Compute number of region covering addr. See Page::GetRegionNumberForAddress
  // method for more details.
  and_(addr, Page::kPageAlignmentMask);
  shr(addr, Page::kRegionSizeLog2);

  // Set dirty mark for region.
  // Bit tests with a memory operand should be avoided on Intel processors,
  // as they usually have long latency and multiple uops. We load the bit base
  // operand to a register at first and store it back after bit set.
  mov(scratch, Operand(object, Page::kDirtyFlagOffset));
  bts(Operand(scratch), addr);
  mov(Operand(object, Page::kDirtyFlagOffset), scratch);
}
