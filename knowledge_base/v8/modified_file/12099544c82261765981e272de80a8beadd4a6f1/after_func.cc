Node* WasmGraphBuilder::StringConcat(Node* head, CheckForNull head_null_check,
                                     Node* tail, CheckForNull tail_null_check,
                                     wasm::WasmCodePosition position) {
  if (head_null_check == kWithNullCheck) head = AssertNotNull(head, position);
  if (tail_null_check == kWithNullCheck) tail = AssertNotNull(tail, position);
  return gasm_->CallBuiltin(
      Builtin::kStringAdd_CheckNone, Operator::kEliminatable, head, tail,
      LOAD_INSTANCE_FIELD(NativeContext, MachineType::TaggedPointer()));
}
