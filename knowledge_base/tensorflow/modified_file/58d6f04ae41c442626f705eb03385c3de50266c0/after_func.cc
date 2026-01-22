    if (err_msg) {
      *err_msg = ec.message();
    }
    return true;
  }

  llvm::sys::Memory::InvalidateInstructionCache(code_block_.base(),
                                                code_block_.allocatedSize());
  return false;
}

}  // namespace

/*static*/ std::unique_ptr<llvm::TargetMachine>
SimpleOrcJIT::InferTargetMachineForJIT(
    const llvm::TargetOptions& target_options,
    llvm::CodeGenOptLevel opt_level) {
  std::vector<std::string> attrs = DetectMachineAttributes();
  // Default preference is 256-bit vectorization because of the attribute
  // `+prefer-256-bit`. Drop `prefer-256-bit` from the attributes by negation
  // for higher target machine features, for example, avx-512 vectorization.
