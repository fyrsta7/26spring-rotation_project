  }

 private:
  const Shape& GetOutputShape(const HloInstruction* gemm) {
    return gemm->shape().IsTuple() ? gemm->shape().tuple_shapes(0)
                                   : gemm->shape();
  }

  absl::StatusOr<se::DeviceMemoryBase> CreateBuffer(const Shape& shape) {
    return AutotunerUtil::CreateBuffer(*redzone_allocator_, shape,
                                       autotune_config_, rng_state_);
  }

  absl::StatusOr<AutotuneResult> TuneGpuBlasLt(const HloInstruction* gemm,
                                               const GemmConfig& gemm_config) {
    GpuBackendConfig gpu_config =
        gemm->backend_config<GpuBackendConfig>().value();
    const GemmBackendConfig& backend_config = gpu_config.gemm_backend_config();

    bool has_matrix_bias = gemm_config.beta != 0.;

    TF_ASSIGN_OR_RETURN(
        bool has_vector_bias,
        gpublas_lt::EpilogueAddsVectorBias(backend_config.epilogue()));

    TF_ASSIGN_OR_RETURN(
        bool has_aux_output,
        gpublas_lt::EpilogueHasAuxiliaryOutput(backend_config.epilogue()));

    TF_ASSIGN_OR_RETURN(auto epilogue,
                        AsBlasLtEpilogue(backend_config.epilogue()));

    se::DeviceMemoryBase a_scale_buffer, b_scale_buffer, c_scale_buffer,
        d_scale_buffer, d_amax_buffer, bias_buffer, aux_buffer;

    if (has_vector_bias) {
      TF_ASSIGN_OR_RETURN(
          bias_buffer,
          CreateBuffer(gemm->operand(has_matrix_bias ? 3 : 2)->shape()));
    }
    if (has_aux_output) {
      TF_ASSIGN_OR_RETURN(aux_buffer,
                          CreateBuffer(gemm->shape().tuple_shapes(1)));
    }

    TF_ASSIGN_OR_RETURN(auto plan,
                        BlasLt::GetMatmulPlan(stream_, gemm_config, epilogue));

    TF_ASSIGN_OR_RETURN(auto algorithms, plan->GetAlgorithms());

    auto tuned_func = [&](const BlasLt::MatmulAlgorithm& algorithm)
        -> absl::StatusOr<se::blas::ProfileResult> {
      se::OwningScratchAllocator<> scratch_allocator(
