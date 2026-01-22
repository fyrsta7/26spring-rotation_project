bool ShouldOptimizeAsSmallFunction(int bytecode_size, int ticks,
                                   bool any_ic_changed,
                                   bool active_tier_is_turboprop) {
  if (any_ic_changed || bytecode_size >= kMaxBytecodeSizeForEarlyOpt)
    return false;
  // Without turboprop we always allow early optimizations for small functions
  if (!FLAG_turboprop) return true;
  // For turboprop, we only do small function optimizations when tiering up from
  // TP-> TF. We should also scale the ticks, so we optimize small functions
  // when reaching one tick for top tier.
  // TODO(turboprop, mythria): Investigate if small function optimization is
  // required at all and avoid this if possible by changing the heuristics to
  // take function size into account.
  return active_tier_is_turboprop &&
         ticks > FLAG_ticks_scale_factor_for_top_tier;
}
