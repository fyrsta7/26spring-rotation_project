                         absl::Span<const std::string_view> exported) {
  std::unique_ptr<JitCompiler> compiler(
      new JitCompiler(std::move(opts), mlir_module));

  auto status = compiler->ComputeOrdinalsForExportedFunctions(exported);
  if (!status.ok()) return status;

  // Initialize LLVM compiler internals.
  InitializeLlvmCompiler();

  return {std::move(compiler)};
}

static std::function<llvm::Error(llvm::Module*)>
MakeOptimizingTransformerForJit(llvm::TargetMachine* targetMachine) {
  return [targetMachine](llvm::Module* m) -> llvm::Error {
    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;

    llvm::PipelineTuningOptions tuningOptions;
    // Vectorization happens at the MLIR level.
    tuningOptions.LoopVectorization = false;
    llvm::PassBuilder pb(targetMachine, tuningOptions);
