    TF_ASSIGN_OR_RETURN(
        auto timer, se::gpu::GpuTimer::Create(se::gpu::AsGpuStream(stream)));
    TF_RETURN_IF_ERROR(ExecuteKernelOnStream(*matmul_kernel, matmul_args,
                                             launch_dimensions[0], stream));
    if (have_reduction) {
      TF_RETURN_IF_ERROR(ExecuteKernelOnStream(*reduce_kernel, reduce_args,
                                               launch_dimensions[1], stream));
    }
    TF_ASSIGN_OR_RETURN(absl::Duration timer_duration,
                        timer.GetElapsedDuration());
    return std::make_optional(timer_duration);
  }

  StatusOr<std::unique_ptr<Executable>> CompileMatmulWithCublas(
      const HloComputation& original_computation,
      CustomHloRunner& custom_hlo_runner) {
    // Create an unoptimized HLO module which does the same as
    // `original_computation`, but with CuBLAS.
    std::unique_ptr<HloModule> module =
        ExtractComputationIntoNewModule(original_computation);
    VLOG(3) << "Extracted module: " << module->ToString();
    BitcastRemover bitcast_remover;
    TF_RETURN_IF_ERROR(bitcast_remover.Run(module.get()).status());
    VLOG(3) << "Deoptimized module: " << module->ToString();

    DebugOptions options =
        original_computation.parent()->config().debug_options();
