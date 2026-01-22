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
  for (int i = ordered_sizes.size() - 1; i >= 0; i--) {
    const int wg_index = ordered_sizes.size() - 1 - i;
