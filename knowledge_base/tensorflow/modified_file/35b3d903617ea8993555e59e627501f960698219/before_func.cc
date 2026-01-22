    target.addLegalDialect<::mlir::LLVM::LLVMDialect>();
    target.addLegalDialect<::mlir::NVVM::NVVMDialect>();
    // TODO(csigg): Remove once we support replacing non-root ops.
    target.addLegalOp<::mlir::gpu::GPUModuleOp, ::mlir::gpu::ModuleEndOp,
                      ::mlir::gpu::YieldOp>();
    if (failed(mlir::applyFullConversion(m, target, patterns))) {
      signalPassFailure();
    }
  }
};

}  // namespace

Status LowerKernelBodiesToNVVM(mlir::ModuleOp module) {
  // We cannot verify as the signature of the kernel is rewritten.
  ::mlir::PassManager pm(module.getContext(), /*verifyPasses=*/false);
  applyPassManagerCLOptions(pm);

  // Rewrite kernel functions to LLVM IR.
