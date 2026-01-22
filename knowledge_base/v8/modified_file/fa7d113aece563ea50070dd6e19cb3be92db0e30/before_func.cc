void Heap::SetOldGenerationAllocationLimit(intptr_t old_gen_size,
                                           double gc_speed,
                                           double mutator_speed) {
  double factor = HeapGrowingFactor(gc_speed, mutator_speed);

  if (FLAG_trace_gc_verbose) {
    PrintIsolate(isolate_,
                 "Heap growing factor %.1f based on mu=%.3f, speed_ratio=%.f "
                 "(gc=%.f, mutator=%.f)\n",
                 factor, kTargetMutatorUtilization, gc_speed / mutator_speed,
                 gc_speed, mutator_speed);
  }

  // We set the old generation growing factor to 2 to grow the heap slower on
  // memory-constrained devices.
  if (max_old_generation_size_ <= kMaxOldSpaceSizeMediumMemoryDevice) {
    factor = Min(factor, kMaxHeapGrowingFactorMemoryConstrained);
  }

  if (FLAG_stress_compaction ||
      mark_compact_collector()->reduce_memory_footprint_) {
    factor = kMinHeapGrowingFactor;
  }

  old_generation_allocation_limit_ =
      CalculateOldGenerationAllocationLimit(factor, old_gen_size);

  if (FLAG_trace_gc_verbose) {
    PrintIsolate(isolate_, "Grow: old size: %" V8_PTR_PREFIX
                           "d KB, new limit: %" V8_PTR_PREFIX "d KB (%.1f)\n",
                 old_gen_size / KB, old_generation_allocation_limit_ / KB,
                 factor);
  }
}
