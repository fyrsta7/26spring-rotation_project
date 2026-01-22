void WasmTableObject::SetFunctionTablePlaceholder(
    Isolate* isolate, Handle<WasmTableObject> table, int entry_index,
    Handle<WasmInstanceObject> instance, int func_index) {
  // Put (instance, func_index) as a Tuple2 into the entry_index.
  // The {WasmExportedFunction} will be created lazily.
  Handle<Tuple2> tuple = isolate->factory()->NewTuple2(
      instance, Handle<Smi>(Smi::FromInt(func_index), isolate),
      AllocationType::kYoung);
  table->entries().set(entry_index, *tuple);
}
