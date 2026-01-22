  // This instruction is used to enforce ordering at compile time. No code is
  // emitted.
  current_should_compute_bottleneck_time_ = false;
  current_properties_[kBytesAccessedKey] = 0;
  current_properties_.set_output_bytes_accessed(0);
  for (int i = 0; i < add_dependency->operand_count(); ++i) {
    current_properties_.set_operand_bytes_accessed(i, 0);
  }
  current_properties_[kOptimalSecondsKey] = 0;
  return OkStatus();
