  outputs->resize(flat_outputs.size());
  for (int original_index : output_original_indices) {
    (*outputs)[original_index] = std::move(*flat_output_iter);
    ++flat_output_iter;
  }

  return OkStatus();
}

tensorflow::Status GraphExecutor::Extend(const GraphDef& graph) {
  return graph_execution_state_->Extend(graph);
}

StatusOr<std::unique_ptr<GraphExecutor::LoadedClientGraph>>
GraphExecutor::ImportAndCompileClientGraph(
    const GraphExecutor::ClientGraph& client_graph) {
  // Step 1 of loading: Import the client graph from proto to an MLIR module.
  auto import_start_time = absl::Now();
  mlir::DialectRegistry registry;
  RegisterMlirDialect(registry);
  auto context = std::make_unique<mlir::MLIRContext>(registry);
  ASSIGN_OR_RETURN_IN_IMPORT(
      auto module, ImportClientGraphToMlirModule(client_graph, context.get()));
  // TODO(b/278143179): Upload module w/o control flow.
  SymbolUids symbol_uids;
  symbol_uids.tf_symbol_uid = MaybeUploadMlirToXsymbol(module.get());

  auto import_duration = absl::Now() - import_start_time;
  LOG(INFO) << "TFRT finished importing client graph (" << &client_graph
            << "). Took " << absl::ToInt64Milliseconds(import_duration)
            << " ms. Client graph name: " << client_graph.name;

  // Step 2 of loading: Compile the MLIR module from TF dialect to TFRT dialect
  // (in BEF).
  // TODO(b/229261464): Unify the sync and async lowering passes so we do not
  // need this branch.
  auto compile_start_time = absl::Now();
  mlir::OwningOpRef<mlir::ModuleOp> module_with_op_keys;
  std::shared_ptr<ExecutableContext> executable_context = nullptr;

  ModelRuntimeContext model_context(&options_,
                                    options_.compile_options.saved_model_dir,
                                    resource_context_.get());

  if (options_.compile_options.compile_to_sync_tfrt_dialect) {
    if (kernel_registry_ == nullptr) {
      return tensorflow::errors::Internal("Missing kernel registry in MLRT.");
    }
    ASSIGN_OR_RETURN_IN_COMPILE(
        executable_context,
        tfrt::BuildExecutableContext(module.get(), *kernel_registry_));

  } else if (options_.enable_mlrt) {
    if (kernel_registry_ == nullptr) {
      return tensorflow::errors::Internal("Missing kernel registry in MLRT.");
    }

    ASSIGN_OR_RETURN_IN_COMPILE(
        auto bytecode_buffer,
        tensorflow::mlrt_compiler::ConvertTfMlirToBytecode(
            options_.compile_options, fallback_state_, module.get(),
            model_context, &module_with_op_keys));
    mlrt::bc::Executable executable(bytecode_buffer.data());
    auto bytecode_executable =
        std::make_unique<mlrt::LoadedExecutable>(executable, *kernel_registry_);
    executable_context = std::make_shared<ExecutableContext>(
        std::move(bytecode_buffer), std::move(bytecode_executable));
  } else {
    tfrt::BefBuffer bef;
    TF_RETURN_IF_ERROR(tensorflow::ConvertTfMlirToBef(
        options_.compile_options, module.get(), &bef, model_context));
    ASSIGN_OR_RETURN_IN_COMPILE(
        auto bef_file, tfrt::CreateBefFileFromBefBuffer(runtime(), bef));
    executable_context = std::make_shared<ExecutableContext>(
        std::move(bef), std::move(bef_file));
