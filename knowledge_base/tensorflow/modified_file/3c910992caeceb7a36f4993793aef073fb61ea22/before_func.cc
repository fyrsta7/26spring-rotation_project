typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
struct LaunchFusedMatMulOp {
  void operator()(
      OpKernelContext* context, const Tensor& a, const Tensor& b,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair,
      FusedComputationType fusion, const FusedComputationArgs& fusion_args,
      Tensor* output, bool use_autotune);
};

template <typename T>
struct LaunchFusedMatMulOp<CPUDevice, T> {
  void operator()(
      OpKernelContext* context, const Tensor& a, const Tensor& b,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair,
      FusedComputationType fusion, const FusedComputationArgs& fusion_args,
      Tensor* output, bool use_autotune) {
    OP_REQUIRES(context, DataTypeToEnum<T>::value != DT_HALF,
                errors::InvalidArgument("_FusedMatMul doesn't support DT_HALF "
                                        "data type on CPU devices."));
    auto lhs = a.matrix<T>();
    auto rhs = b.matrix<T>();
    auto out = output->matrix<T>();

    auto& d = context->eigen_device<CPUDevice>();

    // Executes Eigen contraction with output kernel wrapped into type erased
    // wrapper to reduce the number of unique template instantiations.
    auto executeWithOutputKernel = [&](auto output_kernel) {
      OutputKernelWrapper output_kernel_wrapper(
          [&output_kernel](
              const ContractionOutputMapper<T, Eigen::Index>& output_mapper,
              const Eigen::TensorContractionParams& params, Eigen::Index i,
              Eigen::Index j, Eigen::Index num_rows, Eigen::Index num_cols) {
            output_kernel(output_mapper, params, i, j, num_rows, num_cols);
          });

      out.device(d) = lhs.contract(rhs, dim_pair, output_kernel_wrapper);
    };

    BiasAddArgs<T> bias_add_args;
    if (BiasAddArgs<T>::IsSupported(fusion)) {
      if (fusion == FusedComputationType::kBiasAddWithLeakyRelu) {
        OP_REQUIRES_OK(context, InitBiasAddArgs(context, &bias_add_args,
                                                &fusion_args.leakyrelu_alpha));
      } else {
        OP_REQUIRES_OK(context, InitBiasAddArgs(context, &bias_add_args));
      }
    }

    switch (fusion) {
      case FusedComputationType::kBiasAdd:
        executeWithOutputKernel(WithBiasAdd<T>(bias_add_args));
        break;
      case FusedComputationType::kBiasAddWithRelu:
        executeWithOutputKernel(WithBiasAddAndRelu<T>(bias_add_args));
        break;
      case FusedComputationType::kBiasAddWithRelu6:
        executeWithOutputKernel(WithBiasAddAndRelu6<T>(bias_add_args));
        break;
      case FusedComputationType::kBiasAddWithElu:
