
  static bool IsPseudoFastMath() {
    string optimization_level;
    TF_CHECK_OK(
        ReadStringFromEnvVar("TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_LEVEL", "",
                             &optimization_level));
    optimization_level = str_util::Uppercase(optimization_level);
    return optimization_level == "TENSOR_CORES_ONLY";
  }

 public:
  // Returns the set of ops that are considered numerically-safe (for execution
  // in fp16) and performance-critical. These ops are always converted to fp16.
  static gtl::FlatSet<string> WhiteList(int cuda_version, int cudnn_version) {
    string to_add, to_remove;
    TF_CHECK_OK(ReadStringFromEnvVar(
        "TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_WHITELIST_ADD", "", &to_add));
    TF_CHECK_OK(ReadStringFromEnvVar(
        "TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_WHITELIST_REMOVE", "",
        &to_remove));

    auto list = gtl::FlatSet<string>{
        "BlockLSTM",
        "BlockLSTMV2",
        "BlockLSTMGrad",
        "BlockLSTMGradV2",
        "Conv2D",
        "Conv2DBackpropFilter",
        "Conv2DBackpropInput",
        "CudnnRNN",
        "CudnnRNNBackprop",
        "CudnnRNNBackpropV2",
        "CudnnRNNBackpropV3",
        "CudnnRNNV2",
        "CudnnRNNV3",
        "Einsum",
        "GRUBlockCell",
        "GRUBlockCellGrad",
        "LSTMBlockCell",
        "LSTMBlockCellGrad",
        // TODO(benbarsdell): Enable these when fast and safe fp16 kernels are
        // available for depthwise convolutions.
        // "DepthwiseConv2dNative",
        // "DepthwiseConv2dNativeBackpropFilter",
        // "DepthwiseConv2dNativeBackpropInput",
        "MatMul",
    };
    if (cuda_version >= 9010) {
      // Fp16 BatchMatMul is slow before CUDA 9.1.
      list.insert("BatchMatMul");
