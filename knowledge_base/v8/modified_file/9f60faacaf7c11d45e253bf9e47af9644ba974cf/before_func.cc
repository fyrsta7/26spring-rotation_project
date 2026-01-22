PipelineWasmCompilationJob::Status
PipelineWasmCompilationJob::ExecuteJobImpl() {
  pipeline_.RunPrintAndVerify("Machine", true);
  if (FLAG_wasm_opt || asmjs_origin_) {
    // WASM compilations must *always* be independent of the isolate.
    Isolate* isolate = nullptr;
    PipelineData* data = &data_;
    PipelineRunScope scope(data, "wasm optimization");
    GraphReducer graph_reducer(scope.zone(), data->graph(),
                               data->mcgraph()->Dead());
    DeadCodeElimination dead_code_elimination(&graph_reducer, data->graph(),
                                              data->common(), scope.zone());
    ValueNumberingReducer value_numbering(scope.zone(), data->graph()->zone());
    MachineOperatorReducer machine_reducer(data->mcgraph(), asmjs_origin_);
    CommonOperatorReducer common_reducer(isolate, &graph_reducer, data->graph(),
                                         data->common(), data->machine(),
                                         scope.zone());
    AddReducer(data, &graph_reducer, &dead_code_elimination);
    AddReducer(data, &graph_reducer, &machine_reducer);
    AddReducer(data, &graph_reducer, &common_reducer);
    AddReducer(data, &graph_reducer, &value_numbering);
    graph_reducer.ReduceGraph();
    pipeline_.RunPrintAndVerify("wasm optimization", true);
  }

  if (data_.node_origins()) {
    data_.node_origins()->RemoveDecorator();
  }

  pipeline_.ComputeScheduledGraph();
  if (!pipeline_.SelectInstructions(&linkage_)) return FAILED;
  pipeline_.AssembleCode(&linkage_);
  return SUCCEEDED;
}
