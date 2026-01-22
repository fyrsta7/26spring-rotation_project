#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/math/math_util.h"

namespace xla {

namespace {

// Get the diagonal blocks of the coefficient matrix
XlaOp DiagonalBlocks(XlaOp a, int64_t block_size) {
  XlaBuilder* builder = a.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(a));
    int ndims = shape.rank();
    int64_t n = ShapeUtil::GetDimension(shape, -1);
    int64_t num_blocks = n / block_size;
    absl::Span<int64_t const> batch_dims = absl::MakeConstSpan(
        shape.dimensions().begin(), shape.dimensions().begin() + (ndims - 2));

    XlaOp diag_blocks;

    // If the coefficient matrix is exactly the block size, we just add a
    // singleton dimension i.e. [..., n, n] -> [..., 1, n, n]
    if (n == block_size) {
      std::vector<int64_t> permutation(ndims);
      std::iota(permutation.begin(), permutation.end(), 1);
      permutation.insert(permutation.end() - 2, 0);
      return Transpose(Broadcast(a, /*broadcast_sizes=*/{1}), permutation);
    }

    // We can grab entire blocks using gather
    if (n > block_size) {
      // Construct the starting indices of the diagonal blocks
      auto start_indices =
          Transpose(Broadcast(Mul(Iota(builder, S32, num_blocks),
                                  ConstantR0<int32_t>(builder, block_size)),
                              /*broadcast_sizes=*/{2}),
                    /*permutation=*/{1, 0});

      // Gather the diagonal blocks
      std::vector<int64_t> slice_sizes(ndims);
      GatherDimensionNumbers dim_numbers;
      for (int i = 0; i < ndims - 2; ++i) {
        dim_numbers.add_offset_dims(i);
        slice_sizes[i] = ShapeUtil::GetDimension(shape, i);
      }
      slice_sizes[ndims - 2] = slice_sizes[ndims - 1] = block_size;
      dim_numbers.add_offset_dims(ndims - 1);
      dim_numbers.add_offset_dims(ndims);
      dim_numbers.add_start_index_map(ndims - 2);
      dim_numbers.add_start_index_map(ndims - 1);
      dim_numbers.set_index_vector_dim(1);
      diag_blocks = Gather(a, start_indices, dim_numbers, slice_sizes);
    }

    // The last block might be smaller than the block size,
    // so we will need to pad it
    if (n % block_size != 0) {
      // Pad with identity matrix.
      auto last_blocks =
          SliceInMinorDims(a, {n - n % block_size, n - n % block_size}, {n, n});
      PaddingConfig config = MakeNoPaddingConfig(ndims);
      int64_t padding = block_size - n % block_size;
      config.mutable_dimensions(ndims - 2)->set_edge_padding_high(padding);
      last_blocks =
          Pad(last_blocks, Zero(builder, shape.element_type()), config);

      auto eye =
          IdentityMatrix(builder, shape.element_type(), padding, padding);
      config = MakeNoPaddingConfig(2);
      config.mutable_dimensions(0)->set_edge_padding_low(n % block_size);
      eye = Pad(eye, Zero(builder, shape.element_type()), config);
      eye = Broadcast(eye, batch_dims);
      last_blocks = ConcatInDim(builder, {last_blocks, eye}, ndims - 1);

      // Add a singleton dimension
      // i.e. [..., block_size, block_size] -> [..., 1, block_size, block_size]
      TF_ASSIGN_OR_RETURN(Shape blocks_shape, builder->GetShape(last_blocks));
      auto shape_dims = blocks_shape.dimensions();
      auto last_blocks_dims = std::vector<int64_t>(ndims);
      std::copy(shape_dims.begin(), shape_dims.end(), last_blocks_dims.begin());
      last_blocks_dims.insert(last_blocks_dims.end() - 2, 1);
      last_blocks = Reshape(last_blocks, last_blocks_dims);
