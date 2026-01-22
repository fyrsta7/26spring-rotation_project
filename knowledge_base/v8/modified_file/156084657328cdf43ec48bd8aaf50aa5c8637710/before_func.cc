int FuzzIt(base::Vector<const uint8_t> data) {
  int deopt_count_before = GetWasmEngine()->GetDeoptsExecutedCount();
  v8_fuzzer::FuzzerSupport* support = v8_fuzzer::FuzzerSupport::Get();
  v8::Isolate* isolate = support->GetIsolate();

  Isolate* i_isolate = reinterpret_cast<Isolate*>(isolate);
  v8::Isolate::Scope isolate_scope(isolate);

  // Clear recursive groups: The fuzzer creates random types in every run. These
  // are saved as recursive groups as part of the type canonicalizer, but types
  // from previous runs just waste memory.
  GetTypeCanonicalizer()->EmptyStorageForTesting();
  i_isolate->heap()->ClearWasmCanonicalRttsForTesting();

  v8::HandleScope handle_scope(isolate);
  v8::Context::Scope context_scope(support->GetContext());

  //  We switch it to synchronous mode to avoid the nondeterminism of background
  //  jobs finishing at random times.
  FlagScope<bool> sync_tier_up_scope(&v8_flags.wasm_sync_tier_up, true);
  // Enable the experimental features we want to fuzz. (Note that
  // EnableExperimentalWasmFeatures only enables staged features.)
  FlagScope<bool> deopt_scope(&v8_flags.wasm_deopt, true);
  FlagScope<bool> inlining_indirect(&v8_flags.wasm_inlining_call_indirect,
                                    true);
  // Make inlining more aggressive.
  FlagScope<bool> ignore_call_counts_scope(
      &v8_flags.wasm_inlining_ignore_call_counts, true);
  FlagScope<size_t> inlining_budget(&v8_flags.wasm_inlining_budget,
                                    v8_flags.wasm_inlining_budget * 5);
  FlagScope<size_t> inlining_size(&v8_flags.wasm_inlining_max_size,
                                  v8_flags.wasm_inlining_max_size * 5);
  FlagScope<size_t> inlining_factor(&v8_flags.wasm_inlining_factor,
                                    v8_flags.wasm_inlining_factor * 5);
  // Force new instruction selection.
  FlagScope<bool> new_isel(
      &v8_flags.turboshaft_wasm_instruction_selection_staged, true);

  EnableExperimentalWasmFeatures(isolate);

  v8::TryCatch try_catch(isolate);
  HandleScope scope(i_isolate);
  AccountingAllocator allocator;
  Zone zone(&allocator, ZONE_NAME);

  std::vector<std::string> callees;
  std::vector<std::string> inlinees;
  base::Vector<const uint8_t> buffer =
      GenerateWasmModuleForDeopt(&zone, data, callees, inlinees);

  testing::SetupIsolateForWasmModule(i_isolate);
  ModuleWireBytes wire_bytes(buffer.begin(), buffer.end());
  auto enabled_features = WasmEnabledFeatures::FromIsolate(i_isolate);
  bool valid = GetWasmEngine()->SyncValidate(
      i_isolate, enabled_features, CompileTimeImportsForFuzzing(), wire_bytes);

  if (v8_flags.wasm_fuzzer_gen_test) {
    GenerateTestCase(i_isolate, wire_bytes, valid);
  }

  ErrorThrower thrower(i_isolate, "WasmFuzzerSyncCompile");
  MaybeHandle<WasmModuleObject> compiled = GetWasmEngine()->SyncCompile(
      i_isolate, enabled_features, CompileTimeImportsForFuzzing(), &thrower,
      wire_bytes);
  if (!valid) {
    FATAL("Generated module should validate, but got: %s\n",
          thrower.error_msg());
  }

  std::vector<ExecutionResult> reference_results = PerformReferenceRun(
      callees, wire_bytes, enabled_features, valid, i_isolate);

  if (reference_results.empty()) {
    // If the first run already included non-determinism, there isn't any value
    // in even compiling it (as this fuzzer focusses on executing deopts).
    // Return -1 to not add this case to the corpus.
    return -1;
  }

  Handle<WasmModuleObject> module_object = compiled.ToHandleChecked();
  Handle<WasmInstanceObject> instance =
      GetWasmEngine()
          ->SyncInstantiate(i_isolate, &thrower, module_object, {}, {})
          .ToHandleChecked();

  Handle<WasmExportedFunction> main_function =
      testing::GetExportedFunction(i_isolate, instance, "main")
          .ToHandleChecked();
  int function_to_optimize =
      main_function->shared()->wasm_exported_function_data()->function_index();
  // As the main function has a fixed signature, it doesn't provide great
  // coverage to always optimize and deopt the main function. Instead by only
  // optimizing an inner wasm function, there can be a large amount of
  // parameters with all kinds of types.
  if (!inlinees.empty() && (data.last() & 1)) {
    function_to_optimize--;
  }

  size_t num_callees = reference_results.size();
  for (uint32_t i = 0; i < num_callees; ++i) {
    auto arguments = base::OwnedVector<Handle<Object>>::New(1);
    arguments[0] = handle(Smi::FromInt(i), i_isolate);
    std::unique_ptr<const char[]> exception;
    int32_t result_value = testing::CallWasmFunctionForTesting(
        i_isolate, instance, "main", arguments.as_vector(), &exception);
    ExecutionResult actual_result;
    if (exception) {
      actual_result = exception.get();
    } else {
      actual_result = result_value;
    }
    if (actual_result != reference_results[i]) {
      std::cerr << "Different results vs. reference run for callee "
                << callees[i] << ": \nReference: " << reference_results[i]
                << "\nActual: " << actual_result << std::endl;
      CHECK_EQ(actual_result, reference_results[i]);
      UNREACHABLE();
    }

    TierUpNowForTesting(i_isolate, instance->trusted_data(i_isolate),
                        function_to_optimize);
  }

  // If no deopt was triggered, return -1 to prevent adding this case to the
  // corpus.
  bool deopt_triggered =
      GetWasmEngine()->GetDeoptsExecutedCount() != deopt_count_before;
  return deopt_triggered ? 0 : -1;
}
