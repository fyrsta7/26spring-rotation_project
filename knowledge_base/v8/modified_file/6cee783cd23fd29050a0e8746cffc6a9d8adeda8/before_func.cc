auto Global::get() const -> Val {
  i::Handle<i::WasmGlobalObject> v8_global = impl(this)->v8_object();
  switch (type()->content()->kind()) {
    case I32:
      return Val(v8_global->GetI32());
    case I64:
      return Val(v8_global->GetI64());
    case F32:
      return Val(v8_global->GetF32());
    case F64:
      return Val(v8_global->GetF64());
    case ANYREF:
    case FUNCREF: {
      StoreImpl* store = impl(this)->store();
      i::HandleScope scope(store->i_isolate());
      return Val(V8RefValueToWasm(store, v8_global->GetRef()));
    }
    default:
      // TODO(wasm+): support new value types
      UNREACHABLE();
  }
}
