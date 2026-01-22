bool PipelineImpl::OptimizeGraph(Linkage* linkage) {
  PipelineData* data = this->data_;

  data->BeginPhaseKind("V8.TFLowering");

  // Type the graph and keep the Typer running such that new nodes get
  // automatically typed when they are created.
  Run<TyperPhase>(data->CreateTyper());
  RunPrintAndVerify(TyperPhase::phase_name());
  Run<TypedLoweringPhase>();
  RunPrintAndVerify(TypedLoweringPhase::phase_name());

  if (data->info()->is_loop_peeling_enabled()) {
    Run<LoopPeelingPhase>();
    RunPrintAndVerify(LoopPeelingPhase::phase_name(), true);
  } else {
    Run<LoopExitEliminationPhase>();
    RunPrintAndVerify(LoopExitEliminationPhase::phase_name(), true);
  }

  if (FLAG_turbo_load_elimination) {
    Run<LoadEliminationPhase>();
    RunPrintAndVerify(LoadEliminationPhase::phase_name());
  }
  data->DeleteTyper();

  if (FLAG_turbo_escape) {
    Run<EscapeAnalysisPhase>();
    if (data->compilation_failed()) {
      info()->AbortOptimization(
          BailoutReason::kCyclicObjectStateDetectedInEscapeAnalysis);
      data->EndPhaseKind();
      return false;
    }
    RunPrintAndVerify(EscapeAnalysisPhase::phase_name());
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

  // Run early optimization pass.
  Run<EarlyOptimizationPhase>();
  RunPrintAndVerify(EarlyOptimizationPhase::phase_name(), true);

  Run<EffectControlLinearizationPhase>();
  RunPrintAndVerify(EffectControlLinearizationPhase::phase_name(), true);

  if (FLAG_turbo_store_elimination) {
    Run<StoreStoreEliminationPhase>();
    RunPrintAndVerify(StoreStoreEliminationPhase::phase_name(), true);
  }

  // Optimize control flow.
  if (FLAG_turbo_cf_optimization) {
    Run<ControlFlowOptimizationPhase>();
    RunPrintAndVerify(ControlFlowOptimizationPhase::phase_name(), true);
  }

  // Optimize memory access and allocation operations.
  Run<MemoryOptimizationPhase>();
  // TODO(jarin, rossberg): Remove UNTYPED once machine typing works.
  RunPrintAndVerify(MemoryOptimizationPhase::phase_name(), true);

  // Lower changes that have been inserted before.
  Run<LateOptimizationPhase>();
  // TODO(jarin, rossberg): Remove UNTYPED once machine typing works.
  RunPrintAndVerify(LateOptimizationPhase::phase_name(), true);

  data->source_positions()->RemoveDecorator();
  if (data->info()->trace_turbo_json_enabled()) {
    data->node_origins()->RemoveDecorator();
  }

  ComputeScheduledGraph();

  return SelectInstructions(linkage);
}
