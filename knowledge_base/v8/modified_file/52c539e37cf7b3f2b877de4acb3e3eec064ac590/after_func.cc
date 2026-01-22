  Node* FieldOffset(const wasm::StructType* type, uint32_t field_index) {
    return IntPtrConstant(wasm::ObjectAccess::ToTagged(
        WasmStruct::kHeaderSize + type->field_offset(field_index)));
  }
