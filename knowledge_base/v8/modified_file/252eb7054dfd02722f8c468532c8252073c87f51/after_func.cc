OptimizationReason RuntimeProfiler::ShouldOptimizeIgnition(
    JSFunction* function, JavaScriptFrame* frame) {
  SharedFunctionInfo* shared = function->shared();
  int ticks = shared->profiler_ticks();

  if (shared->bytecode_array()->Size() > kMaxSizeOptIgnition) {
    return OptimizationReason::kDoNotOptimize;
  }

  if (ticks >= kProfilerTicksBeforeOptimization) {
    int typeinfo, generic, total, type_percentage, generic_percentage;
    GetICCounts(function, &typeinfo, &generic, &total, &type_percentage,
                &generic_percentage);
    if (type_percentage >= FLAG_type_info_threshold) {
      // If this particular function hasn't had any ICs patched for enough
      // ticks, optimize it now.
      return OptimizationReason::kHotAndStable;
    } else if (ticks >= kTicksWhenNotEnoughTypeInfo) {
      return OptimizationReason::kHotWithoutMuchTypeInfo;
    } else {
      if (FLAG_trace_opt_verbose) {
        PrintF("[not yet optimizing ");
        function->PrintName();
        PrintF(", not enough type info: %d/%d (%d%%)]\n", typeinfo, total,
               type_percentage);
      }
      return OptimizationReason::kDoNotOptimize;
    }
  } else if (!any_ic_changed_ &&
             shared->bytecode_array()->Size() < kMaxSizeEarlyOptIgnition) {
    // If no IC was patched since the last tick and this function is very
    // small, optimistically optimize it now.
    int typeinfo, generic, total, type_percentage, generic_percentage;
    GetICCounts(function, &typeinfo, &generic, &total, &type_percentage,
                &generic_percentage);
    if (type_percentage >= FLAG_type_info_threshold) {
      return OptimizationReason::kSmallFunction;
    }
  }
  return OptimizationReason::kDoNotOptimize;
}
