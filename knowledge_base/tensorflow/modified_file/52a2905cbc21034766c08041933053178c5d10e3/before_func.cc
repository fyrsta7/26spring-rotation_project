    // max value accepted.
    selection[0] = std::min(algorithm_id_, nbChoices);
    return 1;
  }

  // Called by TensorRT to report choices it made.
  void reportAlgorithms(const nvinfer1::IAlgorithmContext* const* algoContexts,
                        const nvinfer1::IAlgorithm* const* algoChoices,
                        int32_t nbAlgorithms) noexcept override {
  }  // do nothing
};
#endif

Status Converter::BuildCudaEngine(
    TrtUniquePtrType<nvinfer1::ICudaEngine>* engine, int max_batch_size,
    size_t max_workspace_size_bytes, nvinfer1::IGpuAllocator* allocator,
    TRTInt8Calibrator* calibrator, TrtShapeOptimizationProfile* profiles) {
  tensorflow::profiler::AnnotatedTraceMe activity(
      [&]() {
        return tensorflow::profiler::TraceMeOpOverride("TRTEngineOp",
                                                       "BuildEngine");
      },
      tensorflow::profiler::TraceMeLevel::kInfo);

  VLOG(1) << "Configuring TensorRT builder";
  trt_builder_->setMaxBatchSize(max_batch_size);
  trt_builder_->setGpuAllocator(allocator);

  // Create a network configuration and use it to build a TRT engine.
  TrtUniquePtrType<nvinfer1::IBuilderConfig> builder_config(
      trt_builder_->createBuilderConfig());
  builder_config->setMaxWorkspaceSize(max_workspace_size_bytes);

#if IS_TRT_VERSION_GE(7, 1, 3, 0)
  static int32_t trt_algorithm_id = [] {
    int64 trt_algorithm_id;
    TF_CHECK_OK(tensorflow::ReadInt64FromEnvVar("TF_TRT_FIXED_ALGORITHM_ID",
                                                /*default_val=*/-1,
                                                &trt_algorithm_id));
    return static_cast<int32_t>(trt_algorithm_id);
  }();

  if (trt_algorithm_id >= 0) {
    VLOG(1) << "Forcing TRT algorithm selection to: ID=" << trt_algorithm_id;
    StaticAlgorithmSelector trt_algorithm_selector(trt_algorithm_id);
    builder_config->setAlgorithmSelector(&trt_algorithm_selector);
  }
#endif

  if (precision_mode_ == TrtPrecisionMode::FP16) {
    builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
  } else if (precision_mode_ == TrtPrecisionMode::INT8) {
    builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
    builder_config->setFlag(nvinfer1::BuilderFlag::kINT8);
    if (use_calibration_) {
      builder_config->setInt8Calibrator(calibrator);
    } else {
      builder_config->setInt8Calibrator(nullptr);
    }
  }
  if (!use_implicit_batch_ && profiles) {
    TF_RETURN_IF_ERROR(profiles->ConfigureBuilder(
        trt_builder_.get(), builder_config.get(), network()));
  }

  string precision_mode_str;
  TF_RETURN_IF_ERROR(
      TrtPrecisionModeToName(precision_mode_, &precision_mode_str));
  string trt_network_name = StrCat(
      "TF:", TF_VERSION_STRING, ", ",
      "TRT:", absl::StrJoin(GetLoadedTensorRTVersion(), "."), "-",
      "Precision:", precision_mode_str, ", ", "Calibration:", use_calibration_,
      ", ", "Max-Batch-Size:", max_batch_size, ", ",
      "Max-Workspace-Size:", max_workspace_size_bytes);
  VLOG(1) << "Setting TensorRT network name to " << trt_network_name;
  network()->setName(trt_network_name.c_str());

  VLOG(1) << "Building TensorRT engine";
  if (VLOG_IS_ON(2)) {
    VLOG(2) << "Network inputs";
    int n_inputs = network()->getNbInputs();
    for (int i = 0; i < n_inputs; i++) {
      const ITensorProxyPtr input = network()->getInput(i);
      if (*input) {
        VLOG(2) << "  " << i << " " << input->getName();
      } else {
        VLOG(2) << "Could not find input " << i;
      }
    }
  }
  engine->reset(
      trt_builder_->buildEngineWithConfig(*network(), *builder_config));
