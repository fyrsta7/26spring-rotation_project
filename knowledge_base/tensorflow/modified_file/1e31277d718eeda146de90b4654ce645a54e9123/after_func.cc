  pm.addPass(mlir::createCanonicalizerPass());
  if (options.fuse_fill) {
    pm.addNestedPass<FuncOp>(CreateFuseFillIntoTiledReductionPass());
  }
  pm.addNestedPass<FuncOp>(CreateVectorizeTiledOpsPass());
}

}  // namespace

// -------------------------------------------------------------------------- //
// Assemble a TF JitRt pipeline to lower from Tensorflow dialects to Linalg on
// buffers via progressive lowering to MHLO and Linalg.
// -------------------------------------------------------------------------- //
void CreateTfJitRtPipeline(OpPassManager& pm,
                           const TfJitRtPipelineOptions& options) {
  // Break Tensorflow fused operations into primitive operations before
  // lowering to HLO.
  pm.addNestedPass<FuncOp>(CreateFissionPass());

  // Run shape inference to propagate potentially specialized input shapes.
  pm.addPass(std::make_unique<AddTensorflowProducerVersion>());
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Transform TF operation to HLO.
  pm.addNestedPass<FuncOp>(mlir::mhlo::createLegalizeTFPass());

  if (options.legalize_i1_tensors) {
    // Convert 'i1' tensors into 'i8' tensors.
    pm.addPass(CreateJitRtLegalizeI1TypesPass());
  }

  // Resolve all shape constraints (e.g. broadcast constraints that can be
  // proved statically and changed to const witness) early to allow more
  // efficient broadcast operations moving.
  pm.addNestedPass<FuncOp>(
      CreateSymbolicShapeOptimizationPass(/*constraints_only=*/true));

  // Move up broadcasting operations to allow for more fusion opportunities.
  // Add the broadcast propagation pass first, because it can help to avoid
  // exponential complexity from the EarlyBroadcastInDimOp pattern which is used
  // in the merge assuming ops pass further down.
  pm.addNestedPass<FuncOp>(mlir::mhlo::createMergeAssumingOpsPass());
  pm.addNestedPass<FuncOp>(mlir::mhlo::createBroadcastPropagationPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // After all shape constraints removed and broadcasts moved to the top, try
  // to resolve broadcasts that can be converted to linalg generic operations.
  pm.addNestedPass<FuncOp>(CreateSymbolicShapeOptimizationPass());

  // Group reduction and parallel dimensions of reduction operations and realize
  // them through equivalent 1D or 2D reductions, if possible.
  pm.addNestedPass<FuncOp>(mlir::mhlo::createGroupReductionDimensionsPass());

  // Also, try to simplify reshape operations.
  pm.addNestedPass<FuncOp>(mlir::createReshapeSimplifierPass());

  // Transform HLO operations to Linalg and Standard.
  pm.addNestedPass<FuncOp>(mlir::mhlo::createLegalizeHloToLinalgPass());
  pm.addNestedPass<FuncOp>(
      mlir::mhlo::createLegalizeHloShapeOpsToStandardPass());

  // Lower shape dialect to standard to enable linalg canonicalizations (e.g.
  // use linalg inputs instead of outputs for memref.dim operations).
  pm.addNestedPass<FuncOp>(
      mlir::kernel_gen::transforms::CreateShapeSimplification());
  pm.addNestedPass<FuncOp>(mlir::createShapeToShapeLowering());
  pm.addPass(mlir::createConvertShapeToStandardPass());
  pm.addNestedPass<FuncOp>(mlir::createConvertShapeConstraintsPass());

  // Fuse Linalg on tensors operations.
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::memref::createResolveShapedTypeResultDimsPass());
  // Lower index cast on tensors to tensor.generate.
  pm.addNestedPass<FuncOp>(
      mlir::kernel_gen::transforms::CreateLowerIndexCastPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Add linalg passes to perform fusion, tiling, peeling and vectorization.
  AddLinalgTransformations(pm, options);

  // Bufferize Linalg on tensors program.
  // Always run canonicalizer (which does dead code removal) before bufferizing
  // anything.
  pm.addPass(mlir::createCanonicalizerPass());
  // Now bufferize all the compute operations (hlo + linalg) and func signature.
  pm.addPass(
      mlir::kernel_gen::transforms::CreateComputeOpAndFuncBufferizePass());
  pm.addNestedPass<FuncOp>(
      mlir::kernel_gen::transforms::CreateTiledLoopBufferizePass());
  // Now that all compute operations are converted to standard (as a side effect
  // of bufferizing to memref dialect) we can remove the remaining references
  // to unsigned types.
  pm.addPass(mlir::kernel_gen::transforms::CreateConvertToSignlessPass());
  // Always run CSE and canonicalizer (which does dead code removal) before
  // bufferizing anything.
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());
  // Turn tensor constants into global memrefs.
  // TODO(kramerb): Expose the patterns and add them to the bufferize passes.
  pm.addPass(mlir::arith::createConstantBufferizePass(/*alignment=*/64));
  pm.addPass(mlir::kernel_gen::transforms::CreateFinalBufferizePass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Deallocate all temporary buffers.
  pm.addNestedPass<FuncOp>(mlir::bufferization::createBufferDeallocationPass());

  // Do trivial buffer forwarding across linalg.generic operations.
  pm.addNestedPass<FuncOp>(CreateLinalgTrivialBufferForwardingPass());

  // Remove trivial copy operations.
  pm.addNestedPass<FuncOp>(CreateLinalgTrivialCopyRemovalPass());

  if (options.vectorize) {
