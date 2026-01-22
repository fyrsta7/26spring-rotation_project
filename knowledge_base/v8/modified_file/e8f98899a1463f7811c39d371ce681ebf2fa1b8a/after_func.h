struct V8_EXPORT_PRIVATE WasmModule {
  MOVE_ONLY_NO_DEFAULT_CONSTRUCTOR(WasmModule);

  static const uint32_t kPageSize = 0x10000;    // Page size, 64kb.
  static const uint32_t kMinMemPages = 1;       // Minimum memory size = 64kb

  std::unique_ptr<Zone> signature_zone;
  uint32_t initial_pages = 0;      // initial size of the memory in 64k pages
  uint32_t maximum_pages = 0;      // maximum size of the memory in 64k pages
  bool has_maximum_pages = false;  // true if there is a maximum memory size
  bool has_memory = false;        // true if the memory was defined or imported
  bool mem_export = false;        // true if the memory is exported
  int start_function_index = -1;  // start function, >= 0 if any

  std::vector<WasmGlobal> globals;
  uint32_t globals_size = 0;
  uint32_t num_imported_functions = 0;
  uint32_t num_declared_functions = 0;
  uint32_t num_exported_functions = 0;
  WireBytesRef name = {0, 0};
  // TODO(wasm): Add url here, for spec'ed location information.
  std::vector<FunctionSig*> signatures;
  std::vector<WasmFunction> functions;
  std::vector<WasmDataSegment> data_segments;
  std::vector<WasmIndirectFunctionTable> function_tables;
  std::vector<WasmImport> import_table;
  std::vector<WasmExport> export_table;
  std::vector<WasmException> exceptions;
  std::vector<WasmTableInit> table_inits;

  WasmModule() : WasmModule(nullptr) {}
  WasmModule(std::unique_ptr<Zone> owned);

  ModuleOrigin origin() const { return origin_; }
  void set_origin(ModuleOrigin new_value) { origin_ = new_value; }
  bool is_wasm() const { return origin_ == kWasmOrigin; }
  bool is_asm_js() const { return origin_ == kAsmJsOrigin; }

 private:
  // TODO(kschimpf) - Encapsulate more fields.
  ModuleOrigin origin_ = kWasmOrigin;  // origin of the module
};
