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
#if DEBUG
  if (!v8_flags.mock_arraybuffer_allocator) {
    // To flush out bugs earlier, in DEBUG mode, check that all pages of the
    // memory are accessible by reading and writing one byte on each page.
    // Don't do this if the mock ArrayBuffer allocator is enabled.
    uint8_t* mem_start = instance->memory0_start();
    size_t mem_size = instance->memory0_size();
    for (size_t offset = 0; offset < mem_size; offset += wasm::kWasmPageSize) {
      uint8_t val = mem_start[offset];
      USE(val);
      mem_start[offset] = val;
    }
  }
#endif
}
