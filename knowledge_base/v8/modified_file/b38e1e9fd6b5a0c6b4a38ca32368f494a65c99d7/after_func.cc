void SetInstanceMemory(Tagged<WasmInstanceObject> instance,
                       Tagged<JSArrayBuffer> buffer, int memory_index) {
  DisallowHeapAllocation no_gc;
  const WasmModule* module = instance->module();
  const wasm::WasmMemory& memory = module->memories[memory_index];

  bool is_wasm_module = module->origin == wasm::kWasmOrigin;
  bool use_trap_handler = memory.bounds_checks == wasm::kTrapHandler;
  // Asm.js does not use trap handling.
  CHECK_IMPLIES(use_trap_handler, is_wasm_module);
  // ArrayBuffers allocated for Wasm do always have a BackingStore.
  std::shared_ptr<BackingStore> backing_store = buffer->GetBackingStore();
  CHECK_IMPLIES(is_wasm_module, backing_store);
  CHECK_IMPLIES(is_wasm_module, backing_store->is_wasm_memory());
  // Wasm modules compiled to use the trap handler don't have bounds checks,
  // so they must have a memory that has guard regions.
  CHECK_IMPLIES(use_trap_handler, backing_store->has_guard_regions());

  instance->SetRawMemory(memory_index,
                         reinterpret_cast<uint8_t*>(buffer->backing_store()),
                         buffer->byte_length());
}
