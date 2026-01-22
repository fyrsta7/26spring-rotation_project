
 private:
  ContainerInfo cinfo_;

  mutex mu_;
  SparseTensorsMap* sparse_tensors_map_ TF_PT_GUARDED_BY(mu_);
};

class AddSparseToTensorsMapOp : public SparseTensorAccessingOp {
 public:
  explicit AddSparseToTensorsMapOp(OpKernelConstruction* context)
      : SparseTensorAccessingOp(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor* input_indices;
    const Tensor* input_values;
    const Tensor* input_shape;
    SparseTensorsMap* map;

    OP_REQUIRES_OK(context, context->input("sparse_indices", &input_indices));
    OP_REQUIRES_OK(context, context->input("sparse_values", &input_values));
    OP_REQUIRES_OK(context, context->input("sparse_shape", &input_shape));
    OP_REQUIRES_OK(context, GetMap(context, true /* is_writing */, &map));

    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_indices->shape()),
                errors::InvalidArgument(
                    "Input indices should be a matrix but received shape ",
                    input_indices->shape().DebugString()));

    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_values->shape()),
                errors::InvalidArgument(
                    "Input values should be a vector but received shape ",
                    input_values->shape().DebugString()));

    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_shape->shape()),
                errors::InvalidArgument(
                    "Input shape should be a vector but received shape ",
                    input_shape->shape().DebugString()));

    TensorShape input_shape_object;
    OP_REQUIRES_OK(context,
                   TensorShapeUtils::MakeShape(input_shape->vec<int64>().data(),
                                               input_shape->NumElements(),
                                               &input_shape_object));
