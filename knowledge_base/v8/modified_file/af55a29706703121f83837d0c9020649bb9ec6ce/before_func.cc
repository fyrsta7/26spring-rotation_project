bool Heap::CollectGarbage(GarbageCollector collector,
                          GarbageCollectionReason gc_reason,
                          const char* collector_reason,
                          const v8::GCCallbackFlags gc_callback_flags) {
  // The VM is in the GC state until exiting this function.
  VMState<GC> state(isolate_);
  RuntimeCallTimerScope(isolate(), &RuntimeCallStats::GC);

#ifdef DEBUG
  // Reset the allocation timeout to the GC interval, but make sure to
  // allow at least a few allocations after a collection. The reason
  // for this is that we have a lot of allocation sequences and we
  // assume that a garbage collection will allow the subsequent
  // allocation attempts to go through.
  allocation_timeout_ = Max(6, FLAG_gc_interval);
#endif

  EnsureFillerObjectAtTop();

  if (IsYoungGenerationCollector(collector) &&
      !incremental_marking()->IsStopped()) {
    if (FLAG_trace_incremental_marking) {
      isolate()->PrintWithTimestamp(
          "[IncrementalMarking] Scavenge during marking.\n");
    }
  }

  if (collector == MARK_COMPACTOR && FLAG_incremental_marking &&
      !ShouldFinalizeIncrementalMarking() && !ShouldAbortIncrementalMarking() &&
      !incremental_marking()->IsStopped() &&
      !incremental_marking()->should_hurry() &&
      !incremental_marking()->NeedsFinalization() &&
      !IsCloseToOutOfMemory(new_space_->Capacity())) {
    if (!incremental_marking()->IsComplete() &&
        !mark_compact_collector()->marking_deque()->IsEmpty() &&
        !FLAG_gc_global) {
      if (FLAG_trace_incremental_marking) {
        isolate()->PrintWithTimestamp(
            "[IncrementalMarking] Delaying MarkSweep.\n");
      }
      collector = YoungGenerationCollector();
      collector_reason = "incremental marking delaying mark-sweep";
    }
  }

  bool next_gc_likely_to_collect_more = false;
  size_t committed_memory_before = 0;

  if (collector == MARK_COMPACTOR) {
    committed_memory_before = CommittedOldGenerationMemory();
  }

  {
    tracer()->Start(collector, gc_reason, collector_reason);
    DCHECK(AllowHeapAllocation::IsAllowed());
    DisallowHeapAllocation no_allocation_during_gc;
    GarbageCollectionPrologue();

    {
      HistogramTimer* gc_type_timer = GCTypeTimer(collector);
      HistogramTimerScope histogram_timer_scope(gc_type_timer);
      TRACE_EVENT0("v8", gc_type_timer->name());

      next_gc_likely_to_collect_more =
          PerformGarbageCollection(collector, gc_callback_flags);
    }

    GarbageCollectionEpilogue();
    if (collector == MARK_COMPACTOR && FLAG_track_detached_contexts) {
      isolate()->CheckDetachedContextsAfterGC();
    }

    if (collector == MARK_COMPACTOR) {
      size_t committed_memory_after = CommittedOldGenerationMemory();
      size_t used_memory_after = PromotedSpaceSizeOfObjects();
      MemoryReducer::Event event;
      event.type = MemoryReducer::kMarkCompact;
      event.time_ms = MonotonicallyIncreasingTimeInMs();
      // Trigger one more GC if
      // - this GC decreased committed memory,
      // - there is high fragmentation,
      // - there are live detached contexts.
      event.next_gc_likely_to_collect_more =
          (committed_memory_before > committed_memory_after + MB) ||
          HasHighFragmentation(used_memory_after, committed_memory_after) ||
          (detached_contexts()->length() > 0);
      event.committed_memory = committed_memory_after;
      if (deserialization_complete_) {
        memory_reducer_->NotifyMarkCompact(event);
      }
      memory_pressure_level_.SetValue(MemoryPressureLevel::kNone);
    }

    tracer()->Stop(collector);
  }

  if (collector == MARK_COMPACTOR &&
      (gc_callback_flags & (kGCCallbackFlagForced |
                            kGCCallbackFlagCollectAllAvailableGarbage)) != 0) {
    isolate()->CountUsage(v8::Isolate::kForcedGC);
  }

  // Start incremental marking for the next cycle. The heap snapshot
  // generator needs incremental marking to stay off after it aborted.
  // We do this only for scavenger to avoid a loop where mark-compact
  // causes another mark-compact.
  if (IsYoungGenerationCollector(collector) &&
      !ShouldAbortIncrementalMarking()) {
    StartIncrementalMarkingIfAllocationLimitIsReached(kNoGCFlags,
                                                      kNoGCCallbackFlags);
  }

  return next_gc_likely_to_collect_more;
}
