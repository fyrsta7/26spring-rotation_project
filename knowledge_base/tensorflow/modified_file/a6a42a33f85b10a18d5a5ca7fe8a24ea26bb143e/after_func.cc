  TF_LITE_ENSURE_MSG(context,
                     NumDimensions(op_context.input) <= kTransposeMaxDimensions,
                     "Transpose op only supports 1D-6D input arrays.");
  TF_LITE_ENSURE_TYPES_EQ(context, op_context.input->type,
                          op_context.output->type);

  if (!IsConstantTensor(op_context.perm)) {
    SetTensorToDynamic(op_context.output);
    return kTfLiteOk;
  }
  return ResizeOutputTensor(context, &op_context);
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TransposeContext op_context(context, node);

  // Resize the output tensor if the output tensor is dynamic.
  if (IsDynamicTensor(op_context.output)) {
    TF_LITE_ENSURE_OK(context, ResizeOutputTensor(context, &op_context));
  }

  const int* perm_data = GetTensorData<int32_t>(op_context.perm);
  const int size = op_context.perm->dims->data[0];
  TransposeParams params;
  params.perm_count = size;
#ifdef TFLITE_KERNEL_USE_XNNPACK
  xnn_status status;
  CpuBackendContext* cpu_backend_context =
      CpuBackendContext::GetFromContext(context);
  pthreadpool_t threadpool = cpu_backend_context->get_xnnpack_threadpool();
  std::array<size_t, kTransposeMaxDimensions> xnn_input_shape;
  std::array<size_t, kTransposeMaxDimensions> xnn_perm;
  TfLiteIntArray* input_shape = op_context.input->dims;
  for (int i = 0; i < size; ++i) {
    params.perm[i] = perm_data[i];
    xnn_perm[i] = perm_data[i];
    xnn_input_shape[i] = input_shape->data[i];
  }

#else   // TFLITE_KERNEL_USE_XNNPACK
  for (int i = 0; i < size; ++i) {
    params.perm[i] = perm_data[i];
  }
#endif  // TFLITE_KERNEL_USE_XNNPACK
#define TF_LITE_TRANSPOSE(type, scalar)                     \
  type::Transpose(params, GetTensorShape(op_context.input), \
                  GetTensorData<scalar>(op_context.input),  \
                  GetTensorShape(op_context.output),        \
                  GetTensorData<scalar>(op_context.output))

  // Transpose kernel only does rearranging values not numeric evaluations on
  // each cell. It's safe to implement per size of scalar type and this trick
  // keeps the total code size in a reasonable range.
  switch (op_context.input->type) {
    case kTfLiteFloat32:
    case kTfLiteInt32:
      if (kernel_type == kGenericOptimized) {
#ifdef TFLITE_KERNEL_USE_XNNPACK
        status = xnn_run_transpose_nd_x32(
            GetTensorData<int32_t>(op_context.input),
            GetTensorData<int32_t>(op_context.output), size,
            xnn_input_shape.data(), xnn_perm.data(),
            /*flags=*/XNN_FLAG_YIELD_WORKERS, threadpool);
        if (status != xnn_status_success) {
          TF_LITE_KERNEL_LOG(context, "Failed to run xnnpack transpose");
        }
#else   // TFLITE_KERNEL_USE_XNNPACK
        TF_LITE_TRANSPOSE(optimized_ops, int32_t);
#endif  // TFLITE_KERNEL_USE_XNNPACK
      } else {
        TF_LITE_TRANSPOSE(reference_ops, int32_t);
      }
      break;
    case kTfLiteBool:
      if (sizeof(bool) != 1) {
        TF_LITE_TRANSPOSE(reference_ops, bool);
        break;
      }
      [[fallthrough]];
    case kTfLiteUInt8:
    case kTfLiteInt8:
      if (kernel_type == kGenericOptimized) {
#ifdef TFLITE_KERNEL_USE_XNNPACK
        status = xnn_run_transpose_nd_x8(
            GetTensorData<int8_t>(op_context.input),
            GetTensorData<int8_t>(op_context.output), size,
            xnn_input_shape.data(), xnn_perm.data(),
            /*flags=*/XNN_FLAG_YIELD_WORKERS, threadpool);
        if (status != xnn_status_success) {
          TF_LITE_KERNEL_LOG(context, "Failed to run xnnpack transpose");
        }
#else   // TFLITE_KERNEL_USE_XNNPACK
        TF_LITE_TRANSPOSE(optimized_ops, int8_t);
#endif  // TFLITE_KERNEL_USE_XNNPACK
      } else {
        TF_LITE_TRANSPOSE(reference_ops, int8_t);
      }
      break;
    case kTfLiteInt16:
      if (kernel_type == kGenericOptimized) {
#ifdef TFLITE_KERNEL_USE_XNNPACK
        status = xnn_run_transpose_nd_x16(
            GetTensorData<int16_t>(op_context.input),
            GetTensorData<int16_t>(op_context.output), size,
            xnn_input_shape.data(), xnn_perm.data(),
            /*flags=*/XNN_FLAG_YIELD_WORKERS, threadpool);
        if (status != xnn_status_success) {
          TF_LITE_KERNEL_LOG(context, "Failed to run xnnpack transpose");
        }
#else   // TFLITE_KERNEL_USE_XNNPACK
        TF_LITE_TRANSPOSE(optimized_ops, int16_t);
#endif  // TFLITE_KERNEL_USE_XNNPACK
      } else {
        TF_LITE_TRANSPOSE(reference_ops, int16_t);
      }
      break;
