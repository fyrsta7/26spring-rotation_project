CompilationExecutionResult ExecuteCompilationUnits(
    std::weak_ptr<NativeModule> native_module, Counters* counters,
    JobDelegate* delegate, CompileBaselineOnly baseline_only) {
  TRACE_EVENT0("v8.wasm", "wasm.ExecuteCompilationUnits");

  // Execute JS to Wasm wrapper units first, so that they are ready to be
  // finalized by the main thread when the kFinishedBaselineCompilation event is
  // triggered.
  if (ExecuteJSToWasmWrapperCompilationUnits(native_module, delegate) ==
      kYield) {
    return kYield;
  }

  // These fields are initialized in a {BackgroundCompileScope} before
  // starting compilation.
  WasmEngine* engine;
  base::Optional<CompilationEnv> env;
  std::shared_ptr<WireBytesStorage> wire_bytes;
  std::shared_ptr<const WasmModule> module;
  // Task 0 is any main thread (there might be multiple from multiple isolates),
  // worker threads start at 1 (thus the "+ 1").
  STATIC_ASSERT(kMainTaskId == 0);
  int task_id = delegate ? (int{delegate->GetTaskId()} + 1) : kMainTaskId;
  DCHECK_LE(0, task_id);
  CompilationUnitQueues::Queue* queue;
  base::Optional<WasmCompilationUnit> unit;

  WasmFeatures detected_features = WasmFeatures::None();

  // Preparation (synchronized): Initialize the fields above and get the first
  // compilation unit.
  {
    BackgroundCompileScope compile_scope(native_module);
    if (compile_scope.cancelled()) return kYield;
    engine = compile_scope.native_module()->engine();
    env.emplace(compile_scope.native_module()->CreateCompilationEnv());
    wire_bytes = compile_scope.compilation_state()->GetWireBytesStorage();
    module = compile_scope.native_module()->shared_module();
    queue = compile_scope.compilation_state()->GetQueueForCompileTask(task_id);
    unit = compile_scope.compilation_state()->GetNextCompilationUnit(
        queue, baseline_only);
    if (!unit) return kNoMoreUnits;
  }
  TRACE_COMPILE("ExecuteCompilationUnits (task id %d)\n", task_id);

  std::vector<WasmCompilationResult> results_to_publish;
  while (true) {
    ExecutionTier current_tier = unit->tier();
    const char* event_name = GetCompilationEventName(unit.value(), env.value());
    TRACE_EVENT0("v8.wasm", event_name);
    while (unit->tier() == current_tier) {
      // (asynchronous): Execute the compilation.
      WasmCompilationResult result = unit->ExecuteCompilation(
          engine, &env.value(), wire_bytes, counters, &detected_features);
      results_to_publish.emplace_back(std::move(result));

      bool yield = delegate && delegate->ShouldYield();

      // (synchronized): Publish the compilation result and get the next unit.
      BackgroundCompileScope compile_scope(native_module);
      if (compile_scope.cancelled()) return kYield;

      if (!results_to_publish.back().succeeded()) {
        compile_scope.compilation_state()->SetError();
        return kNoMoreUnits;
      }

      // Yield or get next unit.
      if (yield ||
          !(unit = compile_scope.compilation_state()->GetNextCompilationUnit(
                queue, baseline_only))) {
        std::vector<std::unique_ptr<WasmCode>> unpublished_code =
            compile_scope.native_module()->AddCompiledCode(
                VectorOf(std::move(results_to_publish)));
        results_to_publish.clear();
        compile_scope.compilation_state()->SchedulePublishCompilationResults(
            std::move(unpublished_code));
        compile_scope.compilation_state()->OnCompilationStopped(
            detected_features);
        return yield ? kYield : kNoMoreUnits;
      }

      // Publish after finishing a certain amount of units, to avoid contention
      // when all threads publish at the end.
      bool batch_full =
          queue->ShouldPublish(static_cast<int>(results_to_publish.size()));
      // Also publish each time the compilation tier changes from Liftoff to
      // TurboFan, such that we immediately publish the baseline compilation
      // results to start execution, and do not wait for a batch to fill up.
      bool liftoff_finished = unit->tier() != current_tier &&
                              unit->tier() == ExecutionTier::kTurbofan;
      if (batch_full || liftoff_finished) {
        std::vector<std::unique_ptr<WasmCode>> unpublished_code =
            compile_scope.native_module()->AddCompiledCode(
                VectorOf(std::move(results_to_publish)));
        results_to_publish.clear();
        compile_scope.compilation_state()->SchedulePublishCompilationResults(
            std::move(unpublished_code));
      }
    }
  }
  UNREACHABLE();
}
