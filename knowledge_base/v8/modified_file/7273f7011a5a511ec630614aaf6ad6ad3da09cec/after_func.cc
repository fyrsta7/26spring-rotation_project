  void LoadTableSegments(Handle<FixedArray> code_table,
                         Handle<WasmInstanceObject> instance) {
    int function_table_count =
        static_cast<int>(module_->function_tables.size());
    for (int index = 0; index < function_table_count; ++index) {
      WasmIndirectFunctionTable& table = module_->function_tables[index];
      TableInstance& table_instance = table_instances_[index];

      Handle<FixedArray> all_dispatch_tables;
      if (!table_instance.table_object.is_null()) {
        // Get the existing dispatch table(s) with the WebAssembly.Table object.
        all_dispatch_tables = WasmTableObject::AddDispatchTable(
            isolate_, table_instance.table_object,
            Handle<WasmInstanceObject>::null(), index,
            Handle<FixedArray>::null(), Handle<FixedArray>::null());
      }

      // Count the number of table exports for each function (needed for lazy
      // compilation).
      std::unordered_map<uint32_t, uint32_t> num_table_exports;
      if (compile_lazy(module_)) {
        for (auto table_init : module_->table_inits) {
          for (uint32_t func_index : table_init.entries) {
            Code* code =
                Code::cast(code_table->get(static_cast<int>(func_index)));
            // Only increase the counter for lazy compile builtins (it's not
            // needed otherwise).
            if (code->is_wasm_code()) continue;
            DCHECK_EQ(Builtins::kWasmCompileLazy, code->builtin_index());
            ++num_table_exports[func_index];
          }
        }
      }

      // TODO(titzer): this does redundant work if there are multiple tables,
      // since initializations are not sorted by table index.
      for (auto table_init : module_->table_inits) {
        uint32_t base = EvalUint32InitExpr(table_init.offset);
        DCHECK(in_bounds(base, static_cast<uint32_t>(table_init.entries.size()),
                         table_instance.function_table->length()));
        for (int i = 0, e = static_cast<int>(table_init.entries.size()); i < e;
             ++i) {
          uint32_t func_index = table_init.entries[i];
          WasmFunction* function = &module_->functions[func_index];
          int table_index = static_cast<int>(i + base);
          int32_t sig_index = table.map.Find(function->sig);
          DCHECK_GE(sig_index, 0);
          table_instance.signature_table->set(table_index,
                                              Smi::FromInt(sig_index));
          Handle<Code> wasm_code = EnsureTableExportLazyDeoptData(
              isolate_, instance, code_table, func_index,
              table_instance.function_table, table_index, num_table_exports);
          table_instance.function_table->set(table_index, *wasm_code);

          if (!all_dispatch_tables.is_null()) {
            if (js_wrappers_[func_index].is_null()) {
              // No JSFunction entry yet exists for this function. Create one.
              // TODO(titzer): We compile JS->WASM wrappers for functions are
              // not exported but are in an exported table. This should be done
              // at module compile time and cached instead.

              Handle<Code> wrapper_code =
                  js_to_wasm_cache_.CloneOrCompileJSToWasmWrapper(
                      isolate_, module_, wasm_code, func_index);
              MaybeHandle<String> func_name;
              if (module_->origin == kAsmJsOrigin) {
                // For modules arising from asm.js, honor the names section.
                func_name =
                    WasmCompiledModule::ExtractUtf8StringFromModuleBytes(
                        isolate_, compiled_module_, function->name_offset,
                        function->name_length)
                        .ToHandleChecked();
              }
              Handle<WasmExportedFunction> js_function =
                  WasmExportedFunction::New(
                      isolate_, instance, func_name, func_index,
                      static_cast<int>(function->sig->parameter_count()),
                      wrapper_code);
              js_wrappers_[func_index] = js_function;
            }
            table_instance.js_wrappers->set(table_index,
                                            *js_wrappers_[func_index]);

            UpdateDispatchTablesInternal(isolate_, all_dispatch_tables,
                                         table_index, function, wasm_code);
          }
        }
      }

#ifdef DEBUG
      // Check that the count of table exports was accurate. The entries are
      // decremented on each export, so all should be zero now.
      for (auto e : num_table_exports) {
        DCHECK_EQ(0, e.second);
      }
#endif

      // TODO(titzer): we add the new dispatch table at the end to avoid
      // redundant work and also because the new instance is not yet fully
      // initialized.
      if (!table_instance.table_object.is_null()) {
        // Add the new dispatch table to the WebAssembly.Table object.
        all_dispatch_tables = WasmTableObject::AddDispatchTable(
            isolate_, table_instance.table_object, instance, index,
            table_instance.function_table, table_instance.signature_table);
      }
    }
  }
