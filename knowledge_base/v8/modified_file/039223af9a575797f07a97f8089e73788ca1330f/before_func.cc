void StubCache::GenerateProbe(MacroAssembler* masm,
                              Code::Flags flags,
                              Register receiver,
                              Register name,
                              Register scratch,
                              Register extra,
                              Register extra2) {
  Isolate* isolate = masm->isolate();
  Label miss;

  // Make sure that code is valid. The shifting code relies on the
  // entry size being 8.
  ASSERT(sizeof(Entry) == 8);

  // Make sure the flags does not name a specific type.
  ASSERT(Code::ExtractTypeFromFlags(flags) == 0);

  // Make sure that there are no register conflicts.
  ASSERT(!scratch.is(receiver));
  ASSERT(!scratch.is(name));
  ASSERT(!extra.is(receiver));
  ASSERT(!extra.is(name));
  ASSERT(!extra.is(scratch));
  ASSERT(!extra2.is(receiver));
  ASSERT(!extra2.is(name));
  ASSERT(!extra2.is(scratch));
  ASSERT(!extra2.is(extra));

  // Check scratch, extra and extra2 registers are valid.
  ASSERT(!scratch.is(no_reg));
  ASSERT(!extra.is(no_reg));
  ASSERT(!extra2.is(no_reg));

  // Check that the receiver isn't a smi.
  __ JumpIfSmi(receiver, &miss);

  // Get the map of the receiver and compute the hash.
  __ ldr(scratch, FieldMemOperand(name, String::kHashFieldOffset));
  __ ldr(ip, FieldMemOperand(receiver, HeapObject::kMapOffset));
  __ add(scratch, scratch, Operand(ip));
  uint32_t mask = (kPrimaryTableSize - 1) << kHeapObjectTagSize;
  // Mask down the eor argument to the minimum to keep the immediate
  // ARM-encodable.
  __ eor(scratch, scratch, Operand(flags & mask));
  // Prefer ubfx to and_ here because the mask is not ARM-encodable.
  __ Ubfx(scratch, scratch, kHeapObjectTagSize, kPrimaryTableBits);

  // Probe the primary table.
  ProbeTable(isolate,
             masm,
             flags,
             kPrimary,
             name,
             scratch,
             kHeapObjectTagSize,
             extra,
             extra2);

  // Primary miss: Compute hash for secondary probe.
  __ rsb(scratch, name, Operand(scratch, LSL, kHeapObjectTagSize));
  __ add(scratch, scratch, Operand(flags));
  __ Ubfx(scratch, scratch, kHeapObjectTagSize, kSecondaryTableBits);

  // Probe the secondary table.
  ProbeTable(isolate,
             masm,
             flags,
             kSecondary,
             name,
             scratch,
             kHeapObjectTagSize,
             extra,
             extra2);

  // Cache miss: Fall-through and let caller handle the miss by
  // entering the runtime system.
  __ bind(&miss);
}
