  uint32_t consume_element_func_index(WasmModule* module, ValueType expected) {
    WasmFunction* func = nullptr;
    const uint8_t* initial_pc = pc();
    uint32_t index = consume_func_index(module, &func);
    if (tracer_) tracer_->NextLine();
    if (failed()) return index;
    DCHECK_NOT_NULL(func);
    DCHECK_EQ(index, func->func_index);
    ValueType entry_type = ValueType::Ref(func->sig_index);
    if (V8_UNLIKELY(!IsSubtypeOf(entry_type, expected, module))) {
      errorf(initial_pc,
             "Invalid type in element entry: expected %s, got %s instead.",
             expected.name().c_str(), entry_type.name().c_str());
      return index;
    }
    func->declared = true;
    return index;
  }
