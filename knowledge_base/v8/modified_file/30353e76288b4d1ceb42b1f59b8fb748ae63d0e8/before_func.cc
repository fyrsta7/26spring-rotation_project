bool PagedSpaceAllocatorPolicy::EnsureAllocation(int size_in_bytes,
                                                 AllocationAlignment alignment,
                                                 AllocationOrigin origin) {
  if (!allocator_->in_gc() && !(allocator_->identity() == NEW_SPACE &&
                                space_heap()->ShouldOptimizeForLoadTime())) {
    // Start incremental marking before the actual allocation, this allows the
    // allocation function to mark the object black when incremental marking is
    // running.
    space_heap()->StartIncrementalMarkingIfAllocationLimitIsReached(
        allocator_->local_heap(), space_heap()->GCFlagsForIncrementalMarking(),
        kGCCallbackScheduleIdleGarbageCollection);
  }
  if (allocator_->identity() == NEW_SPACE &&
      space_heap()->incremental_marking()->IsStopped()) {
    DCHECK(allocator_->is_main_thread());
    space_heap()->StartMinorMSIncrementalMarkingIfNeeded();
  }

  // We don't know exactly how much filler we need to align until space is
  // allocated, so assume the worst case.
  size_in_bytes += Heap::GetMaximumFillToAlign(alignment);
  if (allocator_->allocation_info().top() + size_in_bytes <=
      allocator_->allocation_info().limit()) {
    return true;
  }
  return RefillLabMain(size_in_bytes, origin);
}
