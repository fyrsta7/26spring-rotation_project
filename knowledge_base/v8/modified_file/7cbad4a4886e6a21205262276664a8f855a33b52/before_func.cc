std::shared_ptr<NativeModule> CompileToNativeModule(
    Isolate* isolate, const WasmFeatures& enabled, ErrorThrower* thrower,
    std::shared_ptr<const WasmModule> module, const ModuleWireBytes& wire_bytes,
    Handle<FixedArray>* export_wrappers_out) {
  const WasmModule* wasm_module = module.get();
  std::shared_ptr<NativeModule> native_module =
      isolate->wasm_engine()->MaybeGetNativeModule(
          wasm_module->origin, wire_bytes.module_bytes(), isolate);
  if (native_module) {
    // TODO(thibaudm): Look into sharing export wrappers.
    CompileJsToWasmWrappers(isolate, wasm_module, export_wrappers_out);
    return native_module;
  }

  TimedHistogramScope wasm_compile_module_time_scope(SELECT_WASM_COUNTER(
      isolate->counters(), wasm_module->origin, wasm_compile, module_time));

  // Embedder usage count for declared shared memories.
  if (wasm_module->has_shared_memory) {
    isolate->CountUsage(v8::Isolate::UseCounterFeature::kWasmSharedMemory);
  }
  OwnedVector<uint8_t> wire_bytes_copy =
      OwnedVector<uint8_t>::Of(wire_bytes.module_bytes());

  // Create a new {NativeModule} first.
  const bool uses_liftoff = module->origin == kWasmOrigin && FLAG_liftoff;
  size_t code_size_estimate =
      wasm::WasmCodeManager::EstimateNativeModuleCodeSize(module.get(),
                                                          uses_liftoff);
  native_module = isolate->wasm_engine()->NewNativeModule(
      isolate, enabled, module, code_size_estimate);
  native_module->SetWireBytes(std::move(wire_bytes_copy));

  CompileNativeModule(isolate, thrower, wasm_module, native_module.get());
  bool cache_hit = !isolate->wasm_engine()->UpdateNativeModuleCache(
      thrower->error(), &native_module, isolate);
  if (thrower->error()) return {};

  if (cache_hit) {
    CompileJsToWasmWrappers(isolate, wasm_module, export_wrappers_out);
    return native_module;
  }
  Impl(native_module->compilation_state())
      ->FinalizeJSToWasmWrappers(isolate, native_module->module(),
                                 export_wrappers_out);

  // Ensure that the code objects are logged before returning.
  isolate->wasm_engine()->LogOutstandingCodesForIsolate(isolate);

  return native_module;
}
