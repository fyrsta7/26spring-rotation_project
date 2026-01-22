  jitrt_flags = new JitRtFlags;
  jitrt_flags->always_specialize = false;
  jitrt_flags->cost_driven_async_parallel_for = false;
  jitrt_flags->vectorize = false;
  jitrt_flag_list = new std::vector<Flag>({
      Flag("always_specialize", &jitrt_flags->always_specialize, ""),
      Flag("cost_driven_async_parallel_for",
           &jitrt_flags->cost_driven_async_parallel_for, ""),
      Flag("vectorize", &jitrt_flags->vectorize, ""),
  });
  xla::ParseFlagsFromEnvAndDieIfUnknown("TF_JITRT_FLAGS", *jitrt_flag_list);
}

void AllocateAndParseFlags() {
  build_ops_flags = new BuildXlaOpsPassFlags;
  build_ops_flags->tf_xla_enable_lazy_compilation = true;
  build_ops_flags->tf_xla_print_cluster_outputs = false;
  build_ops_flags->tf_xla_check_cluster_input_numerics = false;
  build_ops_flags->tf_xla_check_cluster_output_numerics = false;
  build_ops_flags->tf_xla_disable_constant_folding = false;

  mark_for_compilation_flags = new MarkForCompilationPassFlags;
  mark_for_compilation_flags->xla_auto_jit_flag.optimization_level_single_gpu =
      0;
  mark_for_compilation_flags->xla_auto_jit_flag.optimization_level_general = 0;
  mark_for_compilation_flags->tf_xla_min_cluster_size = 4;
  mark_for_compilation_flags->tf_xla_max_cluster_size =
      std::numeric_limits<int32>::max();
  mark_for_compilation_flags->tf_xla_clustering_debug = false;
  mark_for_compilation_flags->tf_xla_cpu_global_jit = false;
  mark_for_compilation_flags->tf_xla_clustering_fuel =
      std::numeric_limits<int64_t>::max();
  mark_for_compilation_flags
      ->tf_xla_disable_deadness_safety_checks_for_debugging = false;
  mark_for_compilation_flags
      ->tf_xla_disable_resource_variable_safety_checks_for_debugging = false;
  mark_for_compilation_flags->tf_xla_deterministic_cluster_names = false;
  mark_for_compilation_flags->tf_xla_persistent_cache_directory = "";

  device_flags = new XlaDeviceFlags;
  device_flags->tf_xla_compile_on_demand = false;
  device_flags->tf_xla_enable_xla_devices = false;

  ops_flags = new XlaOpsCommonFlags;
  ops_flags->tf_xla_always_defer_compilation = false;
  ops_flags->tf_xla_async_compilation = false;

  jitter_flags = new IntroduceFloatingPointJitterPassFlags;
  jitter_flags->jitter_amount = 1e-5;

  // The `enable_mlir_bridge` flag allows the user to explicitly request that
  // their program is (or isn't) compiled using the MLIR-based TF-to-XLA bridge.
  //
  // The `enable_mlir_bridge_is_explicit` variable tracks whether or not the
  // user has made an explicit request. That is, if this variable is set to
  // true, the program honors the user's request as per `enable_mlir_bridge`; if
  // it's set to false, the default behavior is used (which may run either
  // bridge, on a per-graph basis).
  bool enable_mlir_bridge = false;
  bool enable_mlir_bridge_is_explicit = false;
  bool mlir_bridge_safe_mode = false;
  bool enable_mlir_merge_control_flow_pass = true;
  bool enable_mlir_convert_control_to_data_outputs_pass = false;
  auto setter_for_jitter_tensor_names = [](string sequence) {
    jitter_flags->tensor_names = absl::StrSplit(sequence, ',');
    return true;
  };
  // Dump graphs in TFG dialect.
  bool use_tfg_graph_dumper = false;

  flag_list = new std::vector<Flag>(
      {Flag("tf_xla_enable_lazy_compilation",
            &build_ops_flags->tf_xla_enable_lazy_compilation, ""),
       Flag("tf_xla_print_cluster_outputs",
            &build_ops_flags->tf_xla_print_cluster_outputs,
            "If true then insert Print nodes to print out values produced by "
            "XLA clusters."),
       Flag("tf_xla_check_cluster_input_numerics",
            &build_ops_flags->tf_xla_check_cluster_input_numerics,
            "If true then insert CheckNumerics nodes to check all cluster "
            "inputs."),
       Flag("tf_xla_check_cluster_output_numerics",
            &build_ops_flags->tf_xla_check_cluster_output_numerics,
            "If true then insert CheckNumerics nodes to check all cluster "
            "outputs."),
       Flag("tf_xla_disable_constant_folding",
            &build_ops_flags->tf_xla_disable_constant_folding,
            "If true then disables constant folding on TF graph before XLA "
            "compilation."),

       Flag("tf_xla_compile_on_demand", &device_flags->tf_xla_compile_on_demand,
            "Switch a device into 'on-demand' mode, where instead of "
            "autoclustering ops are compiled one by one just-in-time."),

       Flag("tf_xla_enable_xla_devices",
            &device_flags->tf_xla_enable_xla_devices,
            "Generate XLA_* devices, where placing a computation on such a "
            "device"
            "forces compilation by XLA. Deprecated."),

       Flag("tf_xla_always_defer_compilation",
            &ops_flags->tf_xla_always_defer_compilation, ""),
       Flag("tf_xla_async_compilation", &ops_flags->tf_xla_async_compilation,
            "When lazy compilation is enabled, asynchronous compilation starts "
            "the cluster compilation in the background, and the fallback path "
            "is executed until the compilation has finished."),

       Flag("tf_introduce_floating_point_jitter_to_tensors",
            setter_for_jitter_tensor_names, "",
            "The Tensors to add the jitter to.  The tensors are named in the "
            "TensorId format of <node name>:<output idx>."),
       Flag("tf_introduce_floating_point_jitter_amount",
            &jitter_flags->jitter_amount,
            "The amount of jitter to introduce.  This amount is added to each "
            "element in the tensors named in `tensor_names."),

       Flag("tf_mlir_enable_mlir_bridge", &enable_mlir_bridge,
            "Enables experimental MLIR-Based TensorFlow Compiler Bridge.",
            &enable_mlir_bridge_is_explicit),
       Flag("tf_mlir_enable_merge_control_flow_pass",
            &enable_mlir_merge_control_flow_pass,
            "Enables MergeControlFlow pass for MLIR-Based TensorFlow Compiler "
            "Bridge."),
       Flag("tf_mlir_enable_convert_control_to_data_outputs_pass",
            &enable_mlir_convert_control_to_data_outputs_pass,
            "Enables `tf-executor-convert-control-to-data-outputs` pass for "
            "MLIR-Based TensorFlow Compiler Bridge."),
       Flag(
           "tf_mlir_bridge_safe_mode", &mlir_bridge_safe_mode,
           "When tf_mlir_enable_mlir_bridge is true, this field can enable "
           "the MLIR bridge's safe mode. When the MLIR bridge is in safe mode, "
           "it only runs for graphs that use features MLIR bridge currently "
           "supports."),
       Flag("tf_dump_graphs_in_tfg", &use_tfg_graph_dumper,
            "When tf_dump_graphs_in_tfg is true, graphs after transformations "
            "are dumped in MLIR TFG dialect and not in GraphDef")});

  AppendMarkForCompilationPassFlagsInternal(flag_list);
  xla::ParseFlagsFromEnvAndDieIfUnknown("TF_XLA_FLAGS", *flag_list);

  mlir_flags = new MlirCommonFlags;
  if (!enable_mlir_bridge_is_explicit) {
    mlir_flags->tf_mlir_enable_mlir_bridge =
        (mlir_bridge_safe_mode)
            ? ConfigProto::Experimental::
                  MLIR_BRIDGE_ROLLOUT_SAFE_MODE_FALLBACK_ENABLED
            : ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_UNSPECIFIED;
  } else if (enable_mlir_bridge) {
    mlir_flags->tf_mlir_enable_mlir_bridge =
        (mlir_bridge_safe_mode)
            ? ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_SAFE_MODE_ENABLED
            : ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_ENABLED;
  } else {
    mlir_flags->tf_mlir_enable_mlir_bridge =
        ConfigProto::Experimental::MLIR_BRIDGE_ROLLOUT_DISABLED;
