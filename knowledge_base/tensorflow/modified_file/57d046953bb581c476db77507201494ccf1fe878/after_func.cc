#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace {

// GetTensorListDynamicDims collects the dynamic dimensions that a tensorlist
// may carry and returns them in a 2D vector: XlaOp[ElementSize][DimSize]. If a
// dimension is static, a constant dimension is returned. If a dim is dynamic, a
// dynamic XlaOp representing the dynamic size is returned.
StatusOr<std::vector<std::vector<xla::XlaOp>>> GetTensorListDynamicDims(
    XlaOpKernelContext* ctx, const xla::Shape& element_shape,
    const xla::Shape& list_shape, int64_t num_elements) {
  std::vector<int64_t> dynamic_sizes;
  // The multiplier can be a dynamic value.
  TF_RETURN_IF_ERROR(ctx->ConstantInputAsIntVector(0, &dynamic_sizes));
  std::vector<bool> dims_are_dynamic;
  TF_RETURN_IF_ERROR(
      ctx->ResolveInputDynamismIntoPredVector(0, &dims_are_dynamic));
  bool leading_dim_is_dynamic;
  TF_RETURN_IF_ERROR(
      ctx->ResolveInputDynamismIntoPred(1, &leading_dim_is_dynamic));
  std::vector<std::vector<xla::XlaOp>> list_dynamic_dims;
  // Set dynamic dimension size to 0 for initialization value.
  std::vector<xla::XlaOp> dynamic_dims;
  dynamic_dims.reserve(1 + element_shape.dimensions_size());
  if (leading_dim_is_dynamic) {
    dynamic_dims.push_back(ctx->Input(1));
  } else {
    dynamic_dims.push_back(
        xla::ConstantR0<int32>(ctx->builder(), num_elements));
  }
  for (int64_t dim = 0; dim < element_shape.dimensions_size(); ++dim) {
