Node* WasmGraphBuilder::BuildImportCall(const wasm::FunctionSig* sig,
                                        base::Vector<Node*> args,
                                        base::Vector<Node*> rets,
                                        wasm::WasmCodePosition position,
                                        int func_index,
                                        IsReturnCall continuation) {
  return BuildImportCall(sig, args, rets, position,
                         gasm_->Uint32Constant(func_index), continuation);
}
