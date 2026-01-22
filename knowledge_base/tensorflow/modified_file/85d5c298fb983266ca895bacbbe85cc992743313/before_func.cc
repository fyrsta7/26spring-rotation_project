
    mlir::func::FuncOp candidate_func =
        parent_module.lookupSymbol<mlir::func::FuncOp>(renamed_kernel);
    if (!candidate_func) {
      return renamed_kernel;
    }
  }

  return tsl::errors::AlreadyExists(
      absl::StrCat("Could not create a unique function name for op ",
                   op_->getName().getStringRef().str()));
}

tsl::StatusOr<mlir::func::FuncOp> Tf2XlaRewriter::ImportXlaComputation(
    XlaComputation& computation) {
  ModuleOp mlir_module = op_->getParentOfType<ModuleOp>();
  mlir::Builder builder(mlir_module);
  mlir::SymbolTable symbol_table(mlir_module);

  xla::DebugOptions debug_options;
  TF_ASSIGN_OR_RETURN(auto hlo_module_config,
                      xla::HloModule::CreateModuleConfigFromProto(
                          computation.proto(), debug_options));
  TF_ASSIGN_OR_RETURN(
      auto hlo_module,
      xla::HloModule::CreateFromProto(computation.proto(), hlo_module_config));

  std::unordered_map<const xla::HloComputation*, mlir::func::FuncOp>
      function_map;

  TF_ASSIGN_OR_RETURN(FuncOp translated_function,
