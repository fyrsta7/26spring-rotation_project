    }
  }
  entry_func_attrs->set_result_xla_shape(
      func->getAttrOfType<mlir::StringAttr>("result_xla_shape")
          .getValue()
          .str());

  return GpuExecutable::SetUpMlirAllocation(func, buffer_sizes, allocations,
                                            output_info, output_shape);
}

// The order of `thunk_sequence` corresponds to
// `hlo_schedule->ThunkLaunchOrder()`.
Status CompileModuleToLlvmIrImpl(
    HloModule* hlo_module, llvm::LLVMContext* llvm_context,
    const std::string& target_triple, const std::string& data_layout,
    const std::string& platform_name, se::Platform::Id platform_id,
    GpuDeviceInfo gpu_device_info,
    se::CudaComputeCapability cuda_compute_capability,
    se::RocmComputeCapability rocm_compute_capability,
    const HloDataflowAnalysis::CanShareBuffer& can_share_buffer_function,
    int pointer_size, CompileModuleResults* results,
    se::StreamExecutor* stream_exec) {
  results->llvm_module = std::make_unique<llvm::Module>("", *llvm_context);
  results->llvm_module->setTargetTriple(target_triple);
  results->llvm_module->setDataLayout(data_layout);

  TF_RETURN_IF_ERROR(
      ScheduleGpuModule(hlo_module, pointer_size, gpu_device_info));
  {
    HloPassPipeline pipeline("post-scheduling-passes");

    HloPredicate is_nop =
        HloPredicateIsOp<HloOpcode::kParameter, HloOpcode::kConstant,
                         HloOpcode::kBitcast, HloOpcode::kGetTupleElement>;
    pipeline.AddPass<GpuConvertAsyncCollectivesToSync>(is_nop);
    pipeline.AddPass<OptimizationBarrierExpander>();

    TF_RETURN_IF_ERROR(pipeline.Run(hlo_module).status());
  }

  {
    HloPassPipeline pipeline("remat-pipeline");

    HloRematerialization::Options options(
        [pointer_size](const Shape& shape) {
          return GetSizeOfShape(shape, pointer_size);
        },
        // Assume 75% of the total device memory is available for XLA.
        /*memory_limit_bytes=*/gpu_device_info.device_memory_size * 0.75,
        /*block_size_limit=*/1, /*block_rematerialization_factor=*/1,
        /*compact_shape_function=*/nullptr,
        HloRematerialization::RematerializationMode::kRecomputeAndCompress);
    HloRematerialization::RematerializationSizes sizes;
    pipeline.AddPass<HloRematerialization>(options, sizes);

    TF_ASSIGN_OR_RETURN(bool changed, pipeline.Run(hlo_module));
    if (changed) {
      VLOG(1) << "HloRematerialization saved "
              << sizes.before_bytes - sizes.after_bytes << " bytes";
    }
  }

  auto buffer_size_bytes_function =
      [pointer_size](const BufferValue& buffer_value) -> int64_t {
    return GetSizeOfShape(buffer_value.shape(), pointer_size);
  };

  TF_ASSIGN_OR_RETURN(
      results->buffer_assignment,
      BufferAssigner::Run(
          hlo_module,
          std::make_unique<SequentialHloOrdering>(hlo_module->schedule()),
          buffer_size_bytes_function,
          /*color_alignment=*/
          [](LogicalBuffer::Color) { return kXlaAllocatedBufferAlignBytes; },
          /*allocate_buffers_for_constants=*/true,
          /*colorer=*/BufferAssigner::DefaultColorer(),
          /*must_not_live_out=*/{}, can_share_buffer_function));

  VLOG(1) << "Buffer Assignment Stats for " << hlo_module->name() << "\n"
          << results->buffer_assignment->GetStats().ToString();
  DumpHloModuleIfEnabled(*hlo_module, *results->buffer_assignment,
                         absl::StrCat("sm_", cuda_compute_capability.ToString(),
                                      "_gpu_", kAfterOptimizationsDumpName));

  VLOG(1) << "After optimization module fingerprint for " << hlo_module->name()
          << ": " << hlo_module->GetFingerprint128();

  uint64_t start_usecs = tsl::Env::Default()->NowMicros();
  mlir::DialectRegistry registry;
  IrEmitterUnnested::GetDependentDialects(registry);
  auto mlir_context = std::make_unique<mlir::MLIRContext>(registry);
  mlir_context->getDiagEngine().registerHandler(DiagnosticHandler);
  mlir::OwningOpRef<mlir::ModuleOp> mlir_module =
      mlir::ModuleOp::create(mlir::Builder(mlir_context.get()).getUnknownLoc());

  TF_RETURN_IF_ERROR(
      HloToLhloModule(*results->buffer_assignment, *hlo_module, *mlir_module));

  results->module_name =
      mlir::mhlo::GetDebugNameFromLocation(mlir_module->getLoc());

  if (DumpingEnabledForHloModule(*hlo_module)) {
    DumpToFileInDirOrStdout(*hlo_module, "lmhlo", mlir_module.get());
  }

  auto entry_function = mlir::cast<mlir::func::FuncOp>(
      mlir_module->lookupSymbol(hlo_module->entry_computation()->name()));

  TF_RETURN_IF_ERROR(GetMlirAllocationInfo(
      entry_function, &results->allocations, &results->output_info,
      &results->output_shape, &results->entry_func_attrs));

  IrEmitterContext ir_emitter_context(
      hlo_module, /*buffer_assignment=*/nullptr, platform_name, gpu_device_info,
      cuda_compute_capability, rocm_compute_capability, mlir_context.get(),
      results->llvm_module.get());

  ir_emitter_context.set_allocations(results->allocations);

  auto ir_emitter = IrEmitterUnnested::Create(&ir_emitter_context);

  {
    XLA_SCOPED_LOGGING_TIMER(absl::StrCat(
        "GpuCompiler::RunBackend - IR emission for ", hlo_module->name()));

    TF_RETURN_IF_ERROR(ir_emitter->EmitLmhloRegion(&entry_function.getBody()));

    bool supports_runtime_managed_constants =
        // TODO(b/218907125): Implement this feature for ROCm as well.
        platform_id != se::rocm::kROCmPlatformId &&
        hlo_module->config().debug_options().xla_gpu_enable_shared_constants();
    if (supports_runtime_managed_constants) {
      // Remove these globals from the generated code to indicate that XLA is
      // responsible for allocating and initializing them.
      RemoveUnusedAndUninitializedGlobals(ir_emitter_context.llvm_module(),
                                          ir_emitter_context.constants());
    }

    results->constants = std::move(ir_emitter_context.constants());
    uint64_t end_usecs = tsl::Env::Default()->NowMicros();

    // This won't record values for calls that error out (because if they error
    // out we have no way of telling how far through the process we got).
    RecordHloToLlvmDuration(end_usecs - start_usecs);
  }

  // Sizes of all buffers required for running XLA module.
  std::vector<int64_t> buffer_sizes;
  llvm::transform(
      results->allocations, std::back_inserter(buffer_sizes),
      [](const BufferAllocation& allocation) { return allocation.size(); });

  // TODO(ezhulenev): Remove the FP8 check once https://reviews.llvm.org/D140088
  // is submitted. Currently we can't emit LLVM IR with fp8 types.
  if (IsXlaRuntimeExecutableEnabled(hlo_module->config()) &&
      !HasFp8(*hlo_module)) {
    TF_ASSIGN_OR_RETURN(
        results->executable,
        LowerToJitRt(*mlir_module, entry_function.getName(), buffer_sizes,
                     hlo_module->config(), ir_emitter->ConsumeThunkSequence(),
                     /*hlo_module_for_dump=*/hlo_module));
    return OkStatus();
  }

  if (IsOpenXlaRuntimeEnabled(hlo_module->config())) {
    TF_ASSIGN_OR_RETURN(
        results->executable,
