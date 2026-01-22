 public:
  using UseKind = HloInstruction::UseKind;

  // We could rather iterate backwards through fused_instructions_ here, as it
  // is in reverse postorder, and compute whether each fused instruction reuses
  // the value of this parameter, which would save stack space but not allow us
  // to finish early if we find a reuse.
  static UseKind Compute(int64 i, const HloInstruction& hlo) {
    absl::flat_hash_map<const HloInstruction*, UseKind> memoization_cache;
    return ComputeInternal(i, hlo, &memoization_cache);
  }

 private:
  static UseKind ComputeInternal(
      int64 i, const HloInstruction& hlo,
      absl::flat_hash_map<const HloInstruction*, UseKind>* cache) {
    if (auto hlo_param = DynCast<HloParameterInstruction>(&hlo)) {
      if (hlo_param->parameter_number() == i) {
        return UseKind::kUse;
      }
    }

    auto p = cache->emplace(&hlo, UseKind::kNoUse);
    auto value_it = p.first;
    const bool key_is_new = p.second;

    if (key_is_new) {
      for (int64 j = 0; j < hlo.operands_.size(); ++j) {
        UseKind old_val = value_it->second;

        // The next operation invalidates iterators.
        UseKind new_val =
