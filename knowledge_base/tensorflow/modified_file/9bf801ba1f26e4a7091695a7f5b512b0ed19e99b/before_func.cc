// the UnsortedSegmentReductionFunctor kernel:
// 1. Each thread reduces across multiple rows before writing
// answers to the global memory, we can therefore
// write reduction results to global memory less often.
// 2. We may know that the current thread is the only contributor
// to an output element because of the increasing nature of segment
// ids. In such cases, we do not need to use atomic operations
// to write results to global memory.
// In the flattened view of input data (with only outer and inner
// dimension), every thread processes a strip of input data of
// size OuterDimTileSize x 1. This strip runs across multiple
// rows of input data and all reduction elements share one inner
// dimension index.
template <typename T, typename Index, int OuterDimTileSize,
          typename SegmentReductionF, typename SegmentAtomicReductionF>
__global__ void SortedSegmentReductionCustomKernel(
    const Index input_outer_dim_size, const Index inner_dim_size,
    const Index output_outer_dim_size, const Index* __restrict__ segment_ids,
    const T* __restrict__ input, T* __restrict__ output,
    const Index total_stripe_count，const T initial_value) {
  for (int stripe_index : GpuGridRangeX(total_stripe_count)) {
    const Index segment_offset = stripe_index % inner_dim_size;
    const Index input_outer_dim_index_base =
        stripe_index / inner_dim_size * Index(OuterDimTileSize);

    T reduce_res = initial_value;
    Index first_segment_id = segment_ids[input_outer_dim_index_base];
    Index last_output_segment_id = output_outer_dim_size;
    bool is_first_has_race = input_outer_dim_index_base > 0 \
        && first_segment_id == segment_ids[input_outer_dim_index_base - 1] \
        ? true : false;

    const Index actual_stripe_height =
        min(Index(OuterDimTileSize),
            input_outer_dim_size - input_outer_dim_index_base);
    for (Index j = 0; j < actual_stripe_height; j++) {
      Index current_output_segment_id =
          segment_ids[input_outer_dim_index_base + j];
      // Decide whether to write result to global memory.
      // Result is only written to global memory if we move
      // to another segment. Otherwise we can keep accumulating
      // locally.
      if (current_output_segment_id > last_output_segment_id) {
        const Index output_index =
            last_output_segment_id * inner_dim_size + segment_offset;
        // decide whether to write result to global memory using atomic
        // operations
        if (last_output_segment_id == first_segment_id && is_first_has_race) {
          SegmentAtomicReductionF()(output + output_index, reduce_res);
        } else {
          SegmentReductionF()(output + output_index, reduce_res);
        }
        reduce_res = initial_value;
      }
      SegmentReductionF()(&reduce_res,
                          ldg(input + (input_outer_dim_index_base + j) \
                              * inner_dim_size + segment_offset));
      last_output_segment_id = current_output_segment_id;
    }
    // For the last result in a strip, always write using atomic operations
    // due to possible race conditions with threads computing
    // the following strip.
    Index last_input_outer_dim_index =
        input_outer_dim_index_base + actual_stripe_height - 1;
    bool is_last_has_race =
