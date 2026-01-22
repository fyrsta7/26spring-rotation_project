void TurboAssembler::Move(Register dst, Smi source) {
  STATIC_ASSERT(kSmiTag == 0);
  int value = source.value();
  if (value == 0) {
    xorl(dst, dst);
  } else {
    Move(dst, source.ptr(), RelocInfo::NONE);
  }
}
