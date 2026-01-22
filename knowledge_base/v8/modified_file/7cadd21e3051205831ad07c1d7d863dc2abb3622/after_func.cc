bool ShouldOptimizeAsSmallFunction(int bytecode_size, int ticks,
                                   bool any_ic_changed,
                                   bool active_tier_is_turboprop) {
  if (FLAG_turboprop) return false;
  if (any_ic_changed || bytecode_size >= kMaxBytecodeSizeForEarlyOpt)
    return false;
  return true;
}
