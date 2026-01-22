RUNTIME_FUNCTION(Runtime_WasmMemoryGrow) {
  ClearThreadInWasmScope flag_scope(isolate);
  HandleScope scope(isolate);
  DCHECK_EQ(3, args.length());
  Tagged<WasmTrustedInstanceData> trusted_instance_data =
      WasmTrustedInstanceData::cast(args[0]);
  // {memory_index} and {delta_pages} are checked to be positive Smis in the
  // WasmMemoryGrow builtin which calls this runtime function.
  uint32_t memory_index = args.positive_smi_value_at(1);
  uint32_t delta_pages = args.positive_smi_value_at(2);

  Handle<WasmMemoryObject> memory_object{
      trusted_instance_data->memory_object(memory_index), isolate};
  int ret = WasmMemoryObject::Grow(isolate, memory_object, delta_pages);
  // The WasmMemoryGrow builtin which calls this runtime function expects us to
  // always return a Smi.
  DCHECK(!isolate->has_exception());
  return Smi::FromInt(ret);
}
