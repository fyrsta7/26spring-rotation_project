  if (op_type == OperationType::REDUCE_SUM || op_type == OperationType::MEAN) {
    return "((" + a + ") + (" + b + "))";
  } else if (op_type == OperationType::REDUCE_PRODUCT) {
    return "((" + a + ") * (" + b + "))";
  } else if (op_type == OperationType::REDUCE_MAXIMUM) {
    return "max(" + a + ", " + b + ")";
  } else if (op_type == OperationType::REDUCE_MINIMUM) {
    return "min(" + a + ", " + b + ")";
  }
  return "UnsupportedOperation";
}

// max_total_wg_size is pot
int3 GetMaximumPossibleWGSize(const std::vector<int>& ordered_sizes,
                              int max_total_wg_size) {
  int3 wg_size = int3(1, 1, 1);
  int wg_size_total = 1;
  // Make sure that a minimum number of reductions happens inside the loop over
  // reduction dims. Otherwise, the reduction size could equal the number of
  // workgroups and the inner loop would just copy the values to the reducer,
  // which is inefficient.
  const int minimum_loop_reductions = 2;
  int total_loop_reductions = 4;
  for (int i = ordered_sizes.size() - 1; i >= 0; i--) {
    const int wg_index = ordered_sizes.size() - 1 - i;
    if (wg_index >= 3) {
      return wg_size;
    }
    int loop_reductions_dim = 1;
    while (ordered_sizes[i] >= wg_size[wg_index] * 2 * loop_reductions_dim) {
      // Don't increase the work group size of this dim until we have at least
      // 'minimum_loop_reductions' reductions.
      if (total_loop_reductions < minimum_loop_reductions) {
