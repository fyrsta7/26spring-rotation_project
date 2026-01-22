void CompilationStateImpl::InitializeCompilationProgressAfterDeserialization(
    base::Vector<const int> lazy_functions,
    base::Vector<const int> eager_functions) {
  TRACE_EVENT2("v8.wasm", "wasm.CompilationAfterDeserialization",
               "num_lazy_functions", lazy_functions.size(),
               "num_eager_functions", eager_functions.size());
  base::Optional<TimedHistogramScope> lazy_compile_time_scope;
  if (base::TimeTicks::IsHighResolution()) {
    lazy_compile_time_scope.emplace(
        counters()->wasm_compile_after_deserialize());
  }

  auto* module = native_module_->module();
  {
    base::MutexGuard guard(&callbacks_mutex_);
    DCHECK(compilation_progress_.empty());

    // Initialize the compilation progress as if everything was
    // TurboFan-compiled.
    constexpr uint8_t kProgressAfterTurbofanDeserialization =
        RequiredBaselineTierField::encode(ExecutionTier::kTurbofan) |
        RequiredTopTierField::encode(ExecutionTier::kTurbofan) |
        ReachedTierField::encode(ExecutionTier::kTurbofan);
    compilation_progress_.assign(module->num_declared_functions,
                                 kProgressAfterTurbofanDeserialization);

    // Update compilation state for lazy functions.
    constexpr uint8_t kProgressForLazyFunctions =
        RequiredBaselineTierField::encode(ExecutionTier::kNone) |
        RequiredTopTierField::encode(ExecutionTier::kNone) |
        ReachedTierField::encode(ExecutionTier::kNone);
    for (auto func_index : lazy_functions) {
      compilation_progress_[declared_function_index(module, func_index)] =
          kProgressForLazyFunctions;
    }

    // Update compilation state for eagerly compiled functions.
    constexpr bool kNotLazy = false;
    ExecutionTierPair default_tiers =
        GetDefaultTiersPerModule(native_module_, dynamic_tiering_,
                                 native_module_->IsInDebugState(), kNotLazy);
    uint8_t progress_for_eager_functions =
        RequiredBaselineTierField::encode(default_tiers.baseline_tier) |
        RequiredTopTierField::encode(default_tiers.top_tier) |
        ReachedTierField::encode(ExecutionTier::kNone);
    for (auto func_index : eager_functions) {
      // Check that {func_index} is not contained in {lazy_functions}.
      DCHECK_EQ(
          compilation_progress_[declared_function_index(module, func_index)],
          kProgressAfterTurbofanDeserialization);
      compilation_progress_[declared_function_index(module, func_index)] =
          progress_for_eager_functions;
    }
    DCHECK_NE(ExecutionTier::kNone, default_tiers.baseline_tier);
    outstanding_baseline_units_ += eager_functions.size();

    // Export wrappers are compiled synchronously after deserialization, so set
    // that as finished already. Baseline compilation is done if we do not have
    // any Liftoff functions to compile.
    finished_events_.Add(CompilationEvent::kFinishedExportWrappers);
    if (eager_functions.empty() || v8_flags.wasm_lazy_compilation) {
      finished_events_.Add(CompilationEvent::kFinishedBaselineCompilation);
    }
  }
  auto builder = std::make_unique<CompilationUnitBuilder>(native_module_);
  InitializeCompilationUnits(std::move(builder));
  if (!v8_flags.wasm_lazy_compilation) {
    WaitForCompilationEvent(CompilationEvent::kFinishedBaselineCompilation);
  }
}
