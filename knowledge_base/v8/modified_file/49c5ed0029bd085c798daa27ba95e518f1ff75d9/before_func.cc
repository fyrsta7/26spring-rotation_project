void Assembler::add(Register dst, const Operand& src) {
  EnsureSpace ensure_space(this);
  last_pc_ = pc_;
  EMIT(0x03);
  emit_operand(dst, src);
}
