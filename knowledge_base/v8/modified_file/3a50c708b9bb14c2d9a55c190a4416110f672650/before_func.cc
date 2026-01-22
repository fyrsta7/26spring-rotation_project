void Heap::CheckMemoryPressure() {
  if (HighMemoryPressure()) {
    // The optimizing compiler may be unnecessarily holding on to memory.
    isolate()->AbortConcurrentOptimization(BlockingBehavior::kDontBlock);
  }
  MemoryPressureLevel memory_pressure_level = memory_pressure_level_;
  // Reset the memory pressure level to avoid recursive GCs triggered by
  // CheckMemoryPressure from AdjustAmountOfExternalMemory called by
  // the finalizers.
  memory_pressure_level_ = MemoryPressureLevel::kNone;
  if (memory_pressure_level == MemoryPressureLevel::kCritical) {
    TRACE_EVENT0("v8,devtools.timeline", "V8.CheckMemoryPressure");
    CollectGarbageOnMemoryPressure();
  } else if (memory_pressure_level == MemoryPressureLevel::kModerate) {
    if (FLAG_incremental_marking && incremental_marking()->IsStopped()) {
      TRACE_EVENT0("v8,devtools.timeline", "V8.CheckMemoryPressure");
      StartIncrementalMarking(kReduceMemoryFootprintMask,
                              GarbageCollectionReason::kMemoryPressure);
    }
  }
  if (memory_reducer_) {
    MemoryReducer::Event event;
    event.type = MemoryReducer::kPossibleGarbage;
    event.time_ms = MonotonicallyIncreasingTimeInMs();
    memory_reducer_->NotifyPossibleGarbage(event);
  }
}
