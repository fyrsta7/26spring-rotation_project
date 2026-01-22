Node* WasmGraphBuilder::CallIndirect(uint32_t index, Node** args, Node*** rets,
                                     wasm::WasmCodePosition position) {
  DCHECK_NOT_NULL(args[0]);
  DCHECK(module_ && module_->instance);

  MachineOperatorBuilder* machine = jsgraph()->machine();

  // Compute the code object by loading it from the function table.
  Node* key = args[0];

  // Assume only one table for now.
  DCHECK_LE(module_->instance->function_tables.size(), 1u);
  // Bounds check the index.
  uint32_t table_size =
      module_->IsValidTable(0) ? module_->GetTable(0)->max_size : 0;
  wasm::FunctionSig* sig = module_->GetSignature(index);
  if (table_size > 0) {
    // Bounds check against the table size.
    Node* size = Uint32Constant(table_size);
    Node* in_bounds = graph()->NewNode(machine->Uint32LessThan(), key, size);
    trap_->AddTrapIfFalse(wasm::kTrapFuncInvalid, in_bounds, position);
  } else {
    // No function table. Generate a trap and return a constant.
    trap_->AddTrapIfFalse(wasm::kTrapFuncInvalid, Int32Constant(0), position);
    (*rets) = Buffer(sig->return_count());
    for (size_t i = 0; i < sig->return_count(); i++) {
      (*rets)[i] = trap_->GetTrapValue(sig->GetReturn(i));
    }
    return trap_->GetTrapValue(sig);
  }
  Node* table = FunctionTable(0);

  // Load signature from the table and check.
  // The table is a FixedArray; signatures are encoded as SMIs.
  // [sig1, sig2, sig3, ...., code1, code2, code3 ...]
  ElementAccess access = AccessBuilder::ForFixedArrayElement();
  const int fixed_offset = access.header_size - access.tag();
  {
    Node* load_sig = graph()->NewNode(
        machine->Load(MachineType::AnyTagged()), table,
        graph()->NewNode(machine->Int32Add(),
                         graph()->NewNode(machine->Word32Shl(), key,
                                          Int32Constant(kPointerSizeLog2)),
                         Int32Constant(fixed_offset)),
        *effect_, *control_);
    int32_t key = module_->module->function_tables[0].map.Find(sig);
    DCHECK_GE(key, 0);
    Node* sig_match =
        graph()->NewNode(machine->Word32Equal(),
                         BuildChangeSmiToInt32(load_sig), Int32Constant(key));
    trap_->AddTrapIfFalse(wasm::kTrapFuncSigMismatch, sig_match, position);
  }

  // Load code object from the table.
  uint32_t offset = fixed_offset + kPointerSize * table_size;
  Node* load_code = graph()->NewNode(
      machine->Load(MachineType::AnyTagged()), table,
      graph()->NewNode(machine->Int32Add(),
                       graph()->NewNode(machine->Word32Shl(), key,
                                        Int32Constant(kPointerSizeLog2)),
                       Uint32Constant(offset)),
      *effect_, *control_);

  args[0] = load_code;
  return BuildWasmCall(sig, args, rets, position);
}
