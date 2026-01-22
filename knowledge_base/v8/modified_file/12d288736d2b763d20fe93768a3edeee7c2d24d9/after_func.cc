void MacroAssembler::EnsureNotWhite(
    Register value,
    Register bitmap_scratch,
    Register mask_scratch,
    Register load_scratch,
    Label* value_is_white_and_not_data) {
  DCHECK(!AreAliased(value, bitmap_scratch, mask_scratch, t8));
  GetMarkBits(value, bitmap_scratch, mask_scratch);

  // If the value is black or grey we don't need to do anything.
  DCHECK(strcmp(Marking::kWhiteBitPattern, "00") == 0);
  DCHECK(strcmp(Marking::kBlackBitPattern, "10") == 0);
  DCHECK(strcmp(Marking::kGreyBitPattern, "11") == 0);
  DCHECK(strcmp(Marking::kImpossibleBitPattern, "01") == 0);

  Label done;

  // Since both black and grey have a 1 in the first position and white does
  // not have a 1 there we only need to check one bit.
  // Note that we are using a 4-byte aligned 8-byte load.
  if (emit_debug_code()) {
    LoadWordPair(load_scratch,
                 MemOperand(bitmap_scratch, MemoryChunk::kHeaderSize));
  } else {
    lwu(load_scratch, MemOperand(bitmap_scratch, MemoryChunk::kHeaderSize));
  }
  And(t8, mask_scratch, load_scratch);
  Branch(&done, ne, t8, Operand(zero_reg));

  if (emit_debug_code()) {
    // Check for impossible bit pattern.
    Label ok;
    // sll may overflow, making the check conservative.
    dsll(t8, mask_scratch, 1);
    And(t8, load_scratch, t8);
    Branch(&ok, eq, t8, Operand(zero_reg));
    stop("Impossible marking bit pattern");
    bind(&ok);
  }

  // Value is white.  We check whether it is data that doesn't need scanning.
  // Currently only checks for HeapNumber and non-cons strings.
  Register map = load_scratch;  // Holds map while checking type.
  Register length = load_scratch;  // Holds length of object after testing type.
  Label is_data_object;

  // Check for heap-number
  ld(map, FieldMemOperand(value, HeapObject::kMapOffset));
  LoadRoot(t8, Heap::kHeapNumberMapRootIndex);
  {
    Label skip;
    Branch(&skip, ne, t8, Operand(map));
    li(length, HeapNumber::kSize);
    Branch(&is_data_object);
    bind(&skip);
  }

  // Check for strings.
  DCHECK(kIsIndirectStringTag == 1 && kIsIndirectStringMask == 1);
  DCHECK(kNotStringTag == 0x80 && kIsNotStringMask == 0x80);
  // If it's a string and it's not a cons string then it's an object containing
  // no GC pointers.
  Register instance_type = load_scratch;
  lbu(instance_type, FieldMemOperand(map, Map::kInstanceTypeOffset));
  And(t8, instance_type, Operand(kIsIndirectStringMask | kIsNotStringMask));
  Branch(value_is_white_and_not_data, ne, t8, Operand(zero_reg));
  // It's a non-indirect (non-cons and non-slice) string.
  // If it's external, the length is just ExternalString::kSize.
  // Otherwise it's String::kHeaderSize + string->length() * (1 or 2).
  // External strings are the only ones with the kExternalStringTag bit
  // set.
  DCHECK_EQ(0, kSeqStringTag & kExternalStringTag);
  DCHECK_EQ(0, kConsStringTag & kExternalStringTag);
  And(t8, instance_type, Operand(kExternalStringTag));
  {
    Label skip;
    Branch(&skip, eq, t8, Operand(zero_reg));
    li(length, ExternalString::kSize);
    Branch(&is_data_object);
    bind(&skip);
  }

  // Sequential string, either Latin1 or UC16.
  // For Latin1 (char-size of 1) we shift the smi tag away to get the length.
  // For UC16 (char-size of 2) we just leave the smi tag in place, thereby
  // getting the length multiplied by 2.
  DCHECK(kOneByteStringTag == 4 && kStringEncodingMask == 4);
  DCHECK(kSmiTag == 0 && kSmiTagSize == 1);
  lw(t9, UntagSmiFieldMemOperand(value, String::kLengthOffset));
  And(t8, instance_type, Operand(kStringEncodingMask));
  {
    Label skip;
    Branch(&skip, ne, t8, Operand(zero_reg));
    // Adjust length for UC16.
    dsll(t9, t9, 1);
    bind(&skip);
  }
  Daddu(length, t9, Operand(SeqString::kHeaderSize + kObjectAlignmentMask));
  DCHECK(!length.is(t8));
  And(length, length, Operand(~kObjectAlignmentMask));

  bind(&is_data_object);
  // Value is a data object, and it is white.  Mark it black.  Since we know
  // that the object is white we can make it black by flipping one bit.
  lw(t8, MemOperand(bitmap_scratch, MemoryChunk::kHeaderSize));
  Or(t8, t8, Operand(mask_scratch));
  sw(t8, MemOperand(bitmap_scratch, MemoryChunk::kHeaderSize));

  And(bitmap_scratch, bitmap_scratch, Operand(~Page::kPageAlignmentMask));
  lw(t8, MemOperand(bitmap_scratch, MemoryChunk::kLiveBytesOffset));
  Addu(t8, t8, Operand(length));
  sw(t8, MemOperand(bitmap_scratch, MemoryChunk::kLiveBytesOffset));

  bind(&done);
}
