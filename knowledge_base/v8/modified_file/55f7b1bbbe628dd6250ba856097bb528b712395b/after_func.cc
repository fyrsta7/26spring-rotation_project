void JumpTableAssembler::EmitLazyCompileJumpSlot(uint32_t func_index,
                                                 Address lazy_compile_target) {
  // Use a push, because mov to an extended register takes 6 bytes.
  pushq_imm32(func_index);            // 5 bytes
  EmitJumpSlot(lazy_compile_target);  // 5 bytes
}
