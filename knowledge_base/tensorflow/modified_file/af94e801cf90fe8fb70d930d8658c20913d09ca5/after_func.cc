    module->walk([&](mlir::scf::ParallelOp op) {
      unsigned num_loops = op.getNumLoops();
      std::vector<unsigned> combinedLoops;
      combinedLoops.reserve(num_loops);
      for (unsigned i = 0; i < num_loops; ++i) {
        combinedLoops.push_back(i);
      }
      mlir::collapseParallelLoops(op, {combinedLoops});
    });
  }
};
}  // namespace

Status LowerLHLOToGPU(mlir::ModuleOp module, LowerLHLOToGPUOptions options) {
  mlir::PassManager pm(module.getContext());
  applyPassManagerCLOptions(pm);

  // We have to anticipate later unrolling in tiling to make sure that we get
  // the requested tiling after unrolling. Compute the new tiling here if
  // needed.
  llvm::SmallVector<unsigned, 4> tiling_for_unrolling;
  llvm::SmallVector<int64_t, 4> as_int64;
  if (!options.unroll_factors.empty()) {
    tiling_for_unrolling.reserve(options.tile_sizes.size());
    for (auto pair : llvm::zip(options.tile_sizes, options.unroll_factors)) {
      tiling_for_unrolling.push_back(std::get<0>(pair) * std::get<1>(pair));
      as_int64.push_back(std::get<1>(pair));
    }
  } else {
    tiling_for_unrolling.append(options.tile_sizes.begin(),
                                options.tile_sizes.end());
  }

  // Legalize from HLO to LHLO.
  pm.addPass(::mlir::xla_hlo::createLegalizeToLhloPass());
  // Moving `AllocOp`s and inserting missing `DeallocOp`s
  pm.addPass(::mlir::createBufferPlacementPass());
  // Next, we can strip the outer fusion operation.
  pm.addPass(absl::make_unique<FusionOpRemover>());
  // Remove unnecessary LHLO copies.
  pm.addPass(::mlir::xla_lhlo::createLhloCopyRemovalPass());
  // Transform LHLO operations to LinAlg.
  pm.addPass(::mlir::xla_lhlo::createLegalizeLhloToLinalgPass());
  // Fuse linalg operations.
  pm.addPass(::mlir::xla_lhlo::createLhloFuseLinalg(/*use_parallel_loops=*/true,
                                                    tiling_for_unrolling));
  // Legalize reduce operations directly to GPU dialect.
  pm.addPass(::mlir::xla_lhlo::createLegalizeToGpuPass());
  // Transform the Linalg operations inside of the loop nest into parallel
  // loops.
  pm.addPass(::mlir::createConvertLinalgToParallelLoopsPass());
  // Canonicalize the code to simplify index computations. This is needed so
  // that loop bounds have the same value.
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCanonicalizerPass());
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCSEPass());
  // Fuse the inner-most loops.
  pm.addPass(absl::make_unique<FuseInnerParallelLoops>());
  // Run CSE to ensure that loads and stores to the same subview get
  // recognized as such.
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCSEPass());
  // Forward stores to buffers to loads.
  pm.addPass(absl::make_unique<StoreForwardingPass>());
  // Remove now unused temporary buffers.
  pm.addPass(absl::make_unique<DeadTempBufferRemoval>());
  if (!options.unroll_factors.empty()) {
    pm.addPass(::mlir::createParallelLoopTilingPass(as_int64));
  }
  // Project all loop dimensions to X if necessary.
  if (options.collapse_parallel_loops) {
    pm.addPass(absl::make_unique<ParallelLoopCollapsingToFirstDim>());
  }
  // Some basic cleanup.
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCanonicalizerPass());
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCSEPass());
  // Greedily map the remaining loop to GPU hardware dimensions.
  pm.addPass(absl::make_unique<MapParallelLoops>());
  // Apply the mapping.
  pm.addPass(mlir::createParallelLoopToGpuPass());
  // Some basic cleanup.
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCanonicalizerPass());
  pm.addNestedPass<::mlir::FuncOp>(::mlir::createCSEPass());
  // Make loops with min bounds into a conditional plus static bounds.
  // Only do this if we unrolled in the first place.
  if (!options.unroll_factors.empty()) {
    pm.addNestedPass<::mlir::FuncOp>(mlir::createForLoopSpecializationPass());
  }
  // Approximate of requested.
  if (options.use_approximations) {
    pm.addNestedPass<::mlir::FuncOp>(
        ::mlir::xla::createLegalizeTanhToApproximationPass());
  }
  // Move scalar operations into the launch to ensure smaller signatures.
