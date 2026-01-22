    auto blob_or = GetGpuBinaryBlob(gpu_module);
    if (blob_or.ok()) {
      const auto& blob = blob_or.ValueOrDie();
      std::string blob_string(blob.begin(), blob.end());
      gpu_module->setAttr(blob_annotation_,
                          mlir::StringAttr::get(&getContext(), blob_string));
      return;
    }
    // Forward the error by attaching the message to the gpu module.
    gpu_module.emitError(blob_or.status().error_message());
    return signalPassFailure();
  }

  xla::StatusOr<std::vector<uint8_t>> GetGpuBinaryBlob(
      mlir::gpu::GPUModuleOp gpu_module) {
    if (architectures_.empty()) {
      return InternalError("Expected at least one GPU architecture.");
    }

    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(gpu_module, llvmContext);

    if (!llvmModule) {
      return InternalError("Could not translate MLIR module to LLVM IR");
    }

    llvmModule->setModuleIdentifier(gpu_module.getName());

#if TENSORFLOW_USE_ROCM
    xla::HloModuleConfig config;
    xla::DebugOptions options = xla::GetDebugOptionsFromFlags();
    options.set_xla_gpu_ftz(enable_ftz_);
    options.set_xla_gpu_dump_llvmir(print_llvmir_);
    config.set_debug_options(options);

    using AmdGpuHsaco = std::vector<tensorflow::uint8>;
    std::vector<tensorflow::se::HsacoImage> images;
    for (const std::string& arch_str : architectures_) {
      // Parse ROCm architecture.
      absl::string_view consumable_arch(arch_str);
      if (!absl::ConsumePrefix(&consumable_arch, "gfx")) {
        return InternalError(
            "Could not parse ROCm architecture prefix (expected gfx)");
      }
      std::string libdevice_dir = tensorflow::RocdlRoot();
      auto llvm_module_copy = llvm::CloneModule(*llvmModule);
      xla::gpu::GpuVersion gpu_version{arch_str};
      auto hsaco_or = xla::gpu::amdgpu::CompileToHsaco(
          llvm_module_copy.get(), gpu_version, config, libdevice_dir);
      if (!hsaco_or.ok()) {
        return InternalError("Failure when generating HSACO");
      }
      auto hsaco = hsaco_or.ValueOrDie();
      images.push_back({arch_str, std::move(hsaco)});
    }

    // TODO(b/169870789): Revisit the use of fatbins.
    // Bundle HSACO images into a single fatbin.
    return tensorflow::se::BundleGpuAsm(images, tensorflow::RocmRoot());

#elif GOOGLE_CUDA
    llvmModule->setDataLayout(xla::gpu::nvptx::kDataLayout);

    xla::HloModuleConfig config;
    xla::DebugOptions options = xla::GetDebugOptionsFromFlags();
    options.set_xla_gpu_ftz(enable_ftz_);
    // Make sure we use full precision division operations.
    options.set_xla_gpu_dump_llvmir(print_llvmir_);
    (*options.mutable_xla_backend_extra_options())["-nvptx-prec-divf32"] = "2";
    config.set_debug_options(options);

    auto enable_fusion = [](llvm::TargetMachine* target) {
      target->Options.AllowFPOpFusion = llvm::FPOpFusion::FPOpFusionMode::Fast;
    };

    // Compile and collect requested cubin and PTX images.
    std::vector<tensorflow::se::CubinOrPTXImage> images;
    TF_ASSIGN_OR_RETURN(std::string libdevice_dir, GetLibdeviceDir(config));
    auto gpu_asm_opts = xla::gpu::PtxOptsFromConfig(config);
    for (const std::string& arch_str : architectures_) {
      // Parse CUDA architecture.
      absl::string_view consumable_arch(arch_str);
      bool is_compute_profile;
      if (absl::ConsumePrefix(&consumable_arch, "compute_")) {
        is_compute_profile = true;
      } else if (absl::ConsumePrefix(&consumable_arch, "sm_")) {
        is_compute_profile = false;
      } else {
        return InternalError(
            "Could not parse cuda architecture prefix (expected sm_ or "
            "compute_)");
      }
      uint32_t arch;
      if (!absl::SimpleAtoi(consumable_arch, &arch)) {
        return InternalError("Could not parse cuda architecture number");
      }

      int cc_major = arch / 10;
      int cc_minor = arch % 10;
      // Module may be changed by CompileToPtx.
      auto llvm_module_copy = llvm::CloneModule(*llvmModule);
      TF_ASSIGN_OR_RETURN(
          std::string ptx,
          xla::gpu::nvptx::CompileToPtx(
              llvm_module_copy.get(),
              tensorflow::se::CudaComputeCapability{cc_major, cc_minor}, config,
              libdevice_dir, enable_fusion));

      if (print_ptx_) {
        llvm::dbgs() << "Generated PTX code for module '"
                     << gpu_module.getName() << "' on architecture sm_" << arch
                     << ":\n";
        llvm::dbgs() << ptx << "\n";
      }

      TF_ASSIGN_OR_RETURN(std::vector<uint8_t> gpu_asm,
                          tensorflow::se::CompileGpuAsm(
                              cc_major, cc_minor, ptx.c_str(), gpu_asm_opts));

      // Collect cubin (and ptx image if requested).
      images.push_back({absl::StrCat("sm_", arch), std::move(gpu_asm)});
      if (is_compute_profile) {
        std::vector<uint8_t> ptx_bytes;
        std::copy(ptx.begin(), ptx.end(), std::back_inserter(ptx_bytes));
        images.push_back(
