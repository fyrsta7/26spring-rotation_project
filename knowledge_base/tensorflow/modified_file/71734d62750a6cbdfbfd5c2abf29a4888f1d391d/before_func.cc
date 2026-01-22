
  switch (transformation_) {
    case Transformation::kNone:
      scratch_size_ = 0;
      break;
    case Transformation::kF64ToEf57:
      scratch_size_ = sizeof(float) * inner_block_elems_ * inner_block_elems_ *
                      outer_block_elems_a_ * outer_block_elems_b_;
      DCHECK(!inner_kernel_is_memcpy_);
      break;
  }
}

std::vector<int> TransposePlan::ChooseParallelizationStrategy(
    absl::Span<int64_t const> inverse_permutation) {
  std::vector<int> parallelism;
  int available_parallelism = num_threads_requested_;
  parallelism.reserve(loop_order_.size());

  int ndim = permutation_.size();
  const int pos_stride1a = ndim - 1;
  const int pos_stride1b_in_a = permutation_.back();
  // Compute the number of iterations in `loop`.
  auto loop_iterations = [&](const Loop& loop) {
    int a_dim = loop.dim_in_a;
    int b_dim = inverse_permutation[a_dim];
    int64_t tile_size = std::max(a_tiling_[a_dim], b_tiling_[b_dim]);
    int64_t size = loop.tile_interior
                       ? tile_size
                       : (CeilOfRatio(a_dims_[loop.dim_in_a], tile_size));
    if (!inner_kernel_is_memcpy_ && (loop.tile_interior || tile_size == 1)) {
      if (loop.dim_in_a == pos_stride1a) {
        size = CeilOfRatio<int64_t>(size,
                                    inner_block_elems_ * outer_block_elems_a_);
      } else if (loop.dim_in_a == pos_stride1b_in_a) {
        size = CeilOfRatio<int64_t>(size,
                                    inner_block_elems_ * outer_block_elems_b_);
      }
    }
    return size;
  };

  // Estimate the number of bytes each iteration of each loop processes.
  absl::InlinedVector<int64_t, 4> work_in_bytes(loop_order_.size());
  int64_t acc = elem_size_in_bytes_;
  if (!inner_kernel_is_memcpy_) {
    acc *= inner_block_elems_ * inner_block_elems_ * outer_block_elems_a_ *
           outer_block_elems_b_;
  }
  auto work_it = work_in_bytes.rbegin();
  for (auto it = loop_order_.rbegin(); it != loop_order_.rend(); ++it) {
    *work_it++ = acc;
    acc *= loop_iterations(*it);
  }
  VLOG(7) << "Per-loop iteration work in bytes: "
          << absl::StrJoin(work_in_bytes, ",");

  // Heuristic that attempts to parallelize the outermost loops, down to a
  // minimum per-thread number of bytes processed.
  for (size_t i = 0; i < loop_order_.size(); ++i) {
    const Loop& loop = loop_order_[i];
    CHECK_GE(available_parallelism, 1);
    int64_t iterations = loop_iterations(loop);
    int kMinBytesPerThread = inner_kernel_is_memcpy_ ? (1 << 20) : (1 << 17);
    int64_t min_iterations_per_thread =
        CeilOfRatio<int64_t>(kMinBytesPerThread, work_in_bytes[i]);
    int64_t parallel_work = CeilOfRatio(iterations, min_iterations_per_thread);
