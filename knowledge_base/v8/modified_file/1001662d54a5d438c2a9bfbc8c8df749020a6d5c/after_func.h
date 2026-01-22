void TurboAssembler::SmiUntag(Register dst, Register src) {
  DCHECK(dst.Is64Bits() && src.Is64Bits());
  if (FLAG_enable_slow_asserts) {
    AssertSmi(src);
  }
  DCHECK(SmiValuesAre32Bits() || SmiValuesAre31Bits());
  if (COMPRESS_POINTERS_BOOL) {
    Sbfx(dst, src.W(), kSmiShift, kSmiValueSize);
  } else {
    Asr(dst, src, kSmiShift);
  }
}
