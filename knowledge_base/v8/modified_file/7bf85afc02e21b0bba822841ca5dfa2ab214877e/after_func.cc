bool PipelineImpl::OptimizeGraphForMidTier(Linkage* linkage) {
  PipelineData* data = this->data_;

  data->BeginPhaseKind("V8.TFLowering");

  // Type the graph and keep the Typer running such that new nodes get
  // automatically typed when they are created.
  Run<TyperPhase>(data->CreateTyper());
  RunPrintAndVerify(TyperPhase::phase_name());
  Run<TypedLoweringPhase>();
  RunPrintAndVerify(TypedLoweringPhase::phase_name());

  // TODO(9684): Consider rolling this into the preceeding phase or not creating
  // LoopExit nodes at all.
  Run<LoopExitEliminationPhase>();
  RunPrintAndVerify(LoopExitEliminationPhase::phase_name(), true);

  data->DeleteTyper();

  if (FLAG_assert_types) {
    Run<TypeAssertionsPhase>();
    RunPrintAndVerify(TypeAssertionsPhase::phase_name());
  }

  // Perform simplified lowering. This has to run w/o the Typer decorator,
  // because we cannot compute meaningful types anyways, and the computed types
  // might even conflict with the representation/truncation logic.
  Run<SimplifiedLoweringPhase>();
  RunPrintAndVerify(SimplifiedLoweringPhase::phase_name(), true);

  // From now on it is invalid to look at types on the nodes, because the types
  // on the nodes might not make sense after representation selection due to the
  // way we handle truncations; if we'd want to look at types afterwards we'd
  // essentially need to re-type (large portions of) the graph.

  // In order to catch bugs related to type access after this point, we now
  // remove the types from the nodes (currently only in Debug builds).
#ifdef DEBUG
  Run<UntyperPhase>();
  RunPrintAndVerify(UntyperPhase::phase_name(), true);
#endif

  // Run generic lowering pass.
  Run<GenericLoweringPhase>();
  RunPrintAndVerify(GenericLoweringPhase::phase_name(), true);

  data->BeginPhaseKind("V8.TFBlockBuilding");

  ComputeScheduledGraph();

  Run<ScheduledEffectControlLinearizationPhase>();
  RunPrintAndVerify(ScheduledEffectControlLinearizationPhase::phase_name(),
                    true);

  Run<ScheduledMachineLoweringPhase>();
  RunPrintAndVerify(ScheduledMachineLoweringPhase::phase_name(), true);

  // The DecompressionOptimizationPhase updates node's operations but does not
  // otherwise rewrite the graph, thus it is safe to run on a scheduled graph.
  Run<DecompressionOptimizationPhase>();
  RunPrintAndVerify(DecompressionOptimizationPhase::phase_name(), true);

  data->source_positions()->RemoveDecorator();
  if (data->info()->trace_turbo_json()) {
    data->node_origins()->RemoveDecorator();
  }

  return SelectInstructions(linkage);
}
