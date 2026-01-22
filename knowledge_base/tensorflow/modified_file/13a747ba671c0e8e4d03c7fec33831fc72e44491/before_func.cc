  TF::AddGraphExportLoweringPasses(bridge);

  // Run the bridge on the module, in case of failure, the `diag_handler`
  // converts MLIR errors emitted to the MLIRContext into a tensorflow::Status.
  mlir::StatusScopedDiagnosticHandler diag_handler(module.getContext());
  LogicalResult result = bridge.run(module);
  (void)result;
  if (enable_logging || VLOG_IS_ON(1))
    tensorflow::DumpMlirOpToFile("tpu_bridge_after", module);
  return diag_handler.ConsumeStatus();
}
}  // namespace

void CreateTPUBridgePipeline(OpPassManager &pm) {
  // The following ops must be preserved regardless of reachability. Ideally,
  // all graphs should have control dependencies to enforce this but this is
  // currently not the case (see b/177478741).
  const llvm::SmallVector<std::string, 4> ops_to_preserve = {
      "tf.TPUReplicateMetadata", "tf.TPUCompilationResult",
      "tf.TPUReplicatedInput", "tf.TPUReplicatedOutput"};
  pm.addNestedPass<FuncOp>(
      tf_executor::CreateTFExecutorGraphPruningPass(ops_to_preserve));
  // It is assumed at this stage there are no V1 control flow ops as Graph
  // functionalization is ran before import. Ops can be lifted out of
  // tf_executor dialect islands/graphs.
  pm.addNestedPass<FuncOp>(CreateExecutorDialectToFunctionalConversionPass());
  // Run shape inference so that tf_executor/tf_device ops created later will
  // likely to inherit more concrete types.
  pm.addPass(TF::CreateTFShapeInferencePass());
  pm.addNestedPass<FuncOp>(CreateTPUReorderReplicateAndPartitionedInputsPass());
  // Encode this in its own scope so that func_pm is not mistakenly used
  // later on.
  {
    pm.addPass(CreateTPUClusterFormationPass());
    pm.addNestedPass<FuncOp>(TFDevice::CreateDeviceAttributeToLaunchPass());
    OpPassManager &func_pm = pm.nest<FuncOp>();
    // Place DecomposeResourceOpsPass before TFExecutorConstantSinking pass
    // because DecomposeResourceOpsPass uses pattern rewriter which hoists
    // changed constants out of tf_device.Launch.
    func_pm.addPass(TFDevice::CreateDecomposeResourceOpsPass());
    func_pm.addPass(CreateTPUHostComputationExpansionPass());
    func_pm.addPass(CreateTPUUpdateEmbeddingEnqueueOpInputsPass());
  }
  // TODO(b/173622615): Once OutsideCompilation is represented by launch op and
  // the remaining passes including Inliner support it, remove this
  // LaunchToDeviceAttributePass. This LaunchToDeviceAttribute pass needs to
  // come before TPUClusterCleanupAttributes pass or else the device attribute
  // will be removed from launch causing an error.
  pm.addNestedPass<FuncOp>(TFDevice::CreateLaunchToDeviceAttributePass());

  // Note that the region-based control-flow produced here still contains
  // function call ops which get inlined by the subsequent inliner pass.
  pm.addPass(TF::CreateTFFunctionalControlFlowToRegions());
  pm.addPass(mlir::createInlinerPass());
  pm.addNestedPass<FuncOp>(
      TF::CreateDropWhileShapeInvariantInDeviceClusterPass());
  // Run another shape inference pass because resource decomposition might have
  // created new partial types. Also, after dropping `shape_invariant` attribute
  // from While/WhileRegion ops within cluster would lead to more precise
  // shapes.
  pm.addPass(TF::CreateTFShapeInferencePass());
  pm.addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.addPass(CreateTPUClusterCleanupAttributesPass());
  pm.addPass(TFDevice::CreateResourceOpLiftingPass());
  pm.addNestedPass<FuncOp>(createCSEPass());
  pm.addPass(TFDevice::CreateMarkOpsForOutsideCompilationPass());
  pm.addPass(CreateTPUExtractHeadTailOutsideCompilationPass());
  pm.addPass(CreateTPUExtractOutsideCompilationPass());

  pm.addNestedPass<FuncOp>(TFDevice::CreateClusterConstantSinkingPass());
  pm.addPass(TF::CreateResourceDeviceInferencePass());
  pm.addPass(TFDevice::CreateClusterOutliningPass());
