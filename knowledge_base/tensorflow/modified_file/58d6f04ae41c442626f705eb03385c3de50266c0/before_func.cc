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
