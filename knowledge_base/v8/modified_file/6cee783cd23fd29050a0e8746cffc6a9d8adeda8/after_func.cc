auto Global::get() const -> Val {
  i::Handle<i::WasmGlobalObject> v8_global = impl(this)->v8_object();
  switch (v8_global->type()) {
    case i::wasm::kWasmI32:
      return Val(v8_global->GetI32());
    case i::wasm::kWasmI64:
      return Val(v8_global->GetI64());
    case i::wasm::kWasmF32:
      return Val(v8_global->GetF32());
    case i::wasm::kWasmF64:
      return Val(v8_global->GetF64());
    case i::wasm::kWasmAnyRef:
    case i::wasm::kWasmFuncRef: {
      StoreImpl* store = impl(this)->store();
      i::HandleScope scope(store->i_isolate());
      return Val(V8RefValueToWasm(store, v8_global->GetRef()));
    }
    default:
      // TODO(wasm+): support new value types
      UNREACHABLE();
  }
}
