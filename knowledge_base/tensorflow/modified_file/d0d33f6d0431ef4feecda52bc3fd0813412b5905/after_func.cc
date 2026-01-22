
OptimizeDatasetOp::OptimizeDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
  auto& op_name = ctx->def().op();
  if (op_name == kOptimizeDatasetV1) {
    op_version_ = 1;
  } else if (op_name == kOptimizeDatasetV2) {
    op_version_ = 2;
  }
  OP_REQUIRES_OK(ctx,
                 ctx->GetAttr(kOptimizationConfigs, &optimization_configs_));
}

void OptimizeDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                    DatasetBase** output) {
  std::vector<tstring> optimizations;
  if (op_version_ == 1) {
    OP_REQUIRES_OK(
        ctx, ParseVectorArgument<tstring>(ctx, kOptimizations, &optimizations));
  } else if (op_version_ == 2) {
    std::vector<tstring> optimizations_enabled, optimizations_disabled,
        optimizations_default;
    OP_REQUIRES_OK(ctx, ParseVectorArgument<tstring>(ctx, kOptimizationsEnabled,
                                                     &optimizations_enabled));
    OP_REQUIRES_OK(ctx,
                   ParseVectorArgument<tstring>(ctx, kOptimizationsDisabled,
                                                &optimizations_disabled));
    OP_REQUIRES_OK(ctx, ParseVectorArgument<tstring>(ctx, kOptimizationsDefault,
                                                     &optimizations_default));

    string job_name = port::JobName();
    // The map that stores the live experiment names and for how much percentage
    // of the Borg jobs, the experiments will be randomly turned on.
    // clang-format off
    absl::flat_hash_map<string, uint64> live_experiments = {
        {"enable_gradient_descent", 100},
        {"map_parallelization", 20}
    };
    // clang-format on
    auto hash_func = [](const string& str) { return Hash64(str); };
    optimizations = SelectOptimizations(
        job_name, live_experiments, optimizations_enabled,
        optimizations_disabled, optimizations_default, hash_func);

    // Log and record the live experiments that will be applied.
    if (!job_name.empty() && !live_experiments.empty()) {
      VLOG(1) << "The input pipeline is subject to tf.data experiment. "
                 "Please see `go/tf-data-experiments` for more details.";

      for (auto& pair : live_experiments) {
        string experiment = pair.first;
        if (std::find(optimizations.begin(), optimizations.end(), experiment) !=
            optimizations.end()) {
          VLOG(1) << "The live experiment \"" << experiment << "\" is applied.";
          metrics::RecordTFDataExperiment(experiment);
        }
      }
    }
  }

  // The vector stores the graduated experiment names which will be turned on
  // for all input pipelines.
  // clang-format off
  std::vector<string> graduated_experiments = {"disable_intra_op_parallelism"};
  // clang-format on

  // Add the graduated experiments to the optimization list and log them.
  for (auto& experiment : graduated_experiments) {
    if (std::find(optimizations.begin(), optimizations.end(), experiment) ==
        optimizations.end()) {
      optimizations.push_back(experiment);
    }
    VLOG(1) << "The graduated experiment \"" << experiment << "\" is applied.";
  }

  // If there are no optimizations to be applied, directly return the input.
  if (optimizations.empty()) {
    *output = input;
    input->Ref();
    return;
  }

  auto config_factory = [this, &optimizations]() {
    return CreateConfig(optimizations, optimization_configs_);
  };
