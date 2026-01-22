      unnested_hlo, input_shape, use_chain_endings);
  // Fusion inputs with more elements than the reduce op input must participate
  // in non-elementwise operations and we assume that they are not vectorizable
  // for the purpose of estimating the benefit of unrolling. If the kernel is
  // unrolled even with such an assumption,  and the accesses to those inputs
  // turn out to be vectorizable, the compiler will still vectorize them.
  cannot_be_vectorized +=
      NumInputsWithMoreElementsThan(unnested_hlo, input_shape);
  return can_be_vectorized >= cannot_be_vectorized;
}

}  // namespace

ReductionCodegenInfo IrEmitterUnnested::ComputeReductionCodegenInfo(
    const HloInstruction* unnested_hlo, const HloInstruction* first_reduce) {
  const Shape& input_shape = first_reduce->operand(0)->shape();
  ReductionDimensions reduction_dimensions =
      GetReductionKindAndContiguousComponents(*first_reduce);
  VLOG(10) << "is_row_reduction " << reduction_dimensions.is_row_reduction
           << " " << reduction_dimensions.dimensions[0] << " "
           << reduction_dimensions.dimensions[1] << " "
           << reduction_dimensions.dimensions[2];
  auto get_dtype_bits = [](const HloInstruction* i) {
    return primitive_util::BitWidth(i->shape().element_type());
  };

  // For fusion with multiple inputs, use the smallest input dtype to
  // select the reduction_tiling.
  int smallest_input_dtype_bits = get_dtype_bits(first_reduce->operand(0));
  for (xla::HloInstruction* input : unnested_hlo->operands()) {
    smallest_input_dtype_bits =
        std::min(get_dtype_bits(input), smallest_input_dtype_bits);
  }
  std::array<int64, 3> reduction_tiling =
      GetReductionTiling(reduction_dimensions, smallest_input_dtype_bits,
                         &ir_emitter_context_->device_description());
  bool dilated_x =
      reduction_dimensions.is_row_reduction ||
      !IsUnrollingColumnReductionBeneficial(unnested_hlo, input_shape,
                                            reduction_dimensions.dimensions[2]);

  KernelMappingScheme::IndexingOrder indexing_order = [&]() {
    if (reduction_dimensions.is_row_reduction &&
        // Only try to vectorize+coales memory access for rows of even size.
        // For odd row sizes, every other row isn't aligned, so it can't be
        // vectorized.
        reduction_dimensions.dimensions[2] % 2 == 0) {
      return kLinearStridedIndexingX;
    } else if (IsUnrollingColumnReductionBeneficial(
                   unnested_hlo, input_shape,
                   reduction_dimensions.dimensions[2])) {
      return kLinearIndexingX;
    } else {
      return kStridedIndexingX;
    }
  }();
  if (indexing_order == kLinearIndexingX &&
      !reduction_dimensions.is_row_reduction) {
    // Vectorized loads: a single thread reduces two adjacent columns.
    reduction_tiling[2] *= 2;
  }

  int64 num_threads_y = reduction_dimensions.is_row_reduction ? 1 : kWarpSize;
  int64 num_threads_x = [&] {
    if (reduction_dimensions.is_row_reduction) {
      return std::min(
          kWarpSize * kWarpSize,
          RoundUpToNearest(CeilOfRatio(reduction_dimensions.dimensions[2],
                                       reduction_tiling[2]),
                           kWarpSize));
    }
    return kWarpSize;
  }();

  int tile_size_x = reduction_tiling[2] * num_threads_x;

  int vector_size = 1;
  if (indexing_order == kLinearStridedIndexingX) {
    if (reduction_dimensions.dimensions[2] % 2 == 0 &&
        // Assuming XLA will perform the unrolling and LLVM will vectorize,
        // disable the unroll for the cases that LLVM doesn't vectorize.
