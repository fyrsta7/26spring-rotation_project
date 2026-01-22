  void BuildJSToWasmWrapper(bool is_import) {
    const int wasm_count = static_cast<int>(sig_->parameter_count());

    // Build the start and the JS parameter nodes.
    SetEffectControl(Start(wasm_count + 5));

    // Create the js_closure and js_context parameters.
    Node* js_closure =
        graph()->NewNode(mcgraph()->common()->Parameter(
                             Linkage::kJSCallClosureParamIndex, "%closure"),
                         graph()->start());
    Node* js_context = graph()->NewNode(
        mcgraph()->common()->Parameter(
            Linkage::GetJSCallContextParamIndex(wasm_count + 1), "%context"),
        graph()->start());

    // Create the instance_node node to pass as parameter. It is loaded from
    // an actual reference to an instance or a placeholder reference,
    // called {WasmExportedFunction} via the {WasmExportedFunctionData}
    // structure.
    Node* function_data = BuildLoadFunctionDataFromExportedFunction(js_closure);
    instance_node_.set(
        BuildLoadInstanceFromExportedFunctionData(function_data));

    if (!wasm::IsJSCompatibleSignature(sig_, module_, enabled_features_)) {
      // Throw a TypeError. Use the js_context of the calling javascript
      // function (passed as a parameter), such that the generated code is
      // js_context independent.
      BuildCallToRuntimeWithContext(Runtime::kWasmThrowTypeError, js_context,
                                    nullptr, 0);
      TerminateThrow(effect(), control());
      return;
    }

    const int args_count = wasm_count + 1;  // +1 for wasm_code.

    // Check whether the signature of the function allows for a fast
    // transformation. Create a fast transformation path, only if it does.
    bool include_fast_path = QualifiesForFastTransform(sig_);

    // Prepare Param() nodes. Param() nodes can only be created once,
    // so we need to use the same nodes along all possible transformation paths.
    base::SmallVector<Node*, 16> params(args_count);
    for (int i = 0; i < wasm_count; ++i) params[i + 1] = Param(i + 1);

    auto done = gasm_->MakeLabel(MachineRepresentation::kTagged);
    if (include_fast_path) {
      auto slow_path = gasm_->MakeDeferredLabel();
      // Create a condition to determine the transformation path to be used
      // on runtime.
      Node* use_fast_path = gasm_->Int32Constant(1);
      for (int i = 0; i < wasm_count; ++i) {
        Node* can_transform_fast =
            CanTransformFast(params[i + 1], sig_->GetParam(i));
        use_fast_path = gasm_->Word32And(can_transform_fast, use_fast_path);
      }
      gasm_->GotoIfNot(use_fast_path, &slow_path);
      // Convert JS parameters to wasm numbers using the fast transformation
      // and build the call.
      base::SmallVector<Node*, 16> args(args_count);
      for (int i = 0; i < wasm_count; ++i) {
        Node* wasm_param = FromJSFast(params[i + 1], sig_->GetParam(i));
        args[i + 1] = wasm_param;
      }
      Node* jsval =
          BuildCallAndReturn(is_import, js_context, function_data, args);
      gasm_->Goto(&done, jsval);
      gasm_->Bind(&slow_path);
    }
    // Convert JS parameters to wasm numbers using the default transformation
    // and build the call.
    base::SmallVector<Node*, 16> args(args_count);
    for (int i = 0; i < wasm_count; ++i) {
      Node* wasm_param = FromJS(params[i + 1], js_context, sig_->GetParam(i));
      args[i + 1] = wasm_param;
    }
    Node* jsval =
        BuildCallAndReturn(is_import, js_context, function_data, args);
    // If both the default and a fast transformation paths are present,
    // get the return value based on the path used.
    if (include_fast_path) {
      gasm_->Goto(&done, jsval);
      gasm_->Bind(&done);
      Return(done.PhiAt(0));
    } else {
      Return(jsval);
    }
    if (ContainsInt64(sig_)) LowerInt64(kCalledFromJS);
  }
