// Canonicalize operations in functions.
struct TFOptimizePass : public FunctionPass<TFOptimizePass> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    auto func = getFunction();
    populateWithGenerated(&getContext(), &patterns);
    applyPatternsGreedily(func, patterns);
  }
};

}  // namespace

// NOLINTNEXTLINE - MLIR contract is pass by mutable reference.
void CreateTFStandardPipeline(OpPassManager &pm,
                              const StandardPipelineOptions &options) {
  OpPassManager &func_pm = pm.nest<FuncOp>();

  // First operates on the executor dialect:
  // - eliminate trivial switch/merge
  // - fuse islands as much as possible.
  // - materialize the eventual "pass-through" ops by inlining their content.
  func_pm.addPass(tf_executor::CreateSwitchFoldPass());
  func_pm.addPass(tf_executor::CreateTFExecutorIslandCoarseningPass());
