bool PagedSpace::RawSlowRefillLinearAllocationArea(int size_in_bytes) {
  // Allocation in this space has failed.
  DCHECK_GE(size_in_bytes, 0);
  const int kMaxPagesToSweep = 1;

  if (RefillLinearAllocationAreaFromFreeList(size_in_bytes)) return true;

  MarkCompactCollector* collector = heap()->mark_compact_collector();
  // Sweeping is still in progress.
  if (collector->sweeping_in_progress()) {
    if (FLAG_concurrent_sweeping && !is_local() &&
        !collector->sweeper()->AreSweeperTasksRunning()) {
      collector->EnsureSweepingCompleted();
    }

    // First try to refill the free-list, concurrent sweeper threads
    // may have freed some objects in the meantime.
    RefillFreeList();

    // Retry the free list allocation.
    if (RefillLinearAllocationAreaFromFreeList(
            static_cast<size_t>(size_in_bytes)))
      return true;

    // If sweeping is still in progress try to sweep pages.
    int max_freed = collector->sweeper()->ParallelSweepSpace(
        identity(), size_in_bytes, kMaxPagesToSweep);
    RefillFreeList();
    if (max_freed >= size_in_bytes) {
      if (RefillLinearAllocationAreaFromFreeList(
              static_cast<size_t>(size_in_bytes)))
        return true;
    }
  }

  if (is_local()) {
    // The main thread may have acquired all swept pages. Try to steal from
    // it. This can only happen during young generation evacuation.
    PagedSpace* main_space = heap()->paged_space(identity());
    Page* page = main_space->RemovePageSafe(size_in_bytes);
    if (page != nullptr) {
      AddPage(page);
      if (RefillLinearAllocationAreaFromFreeList(
              static_cast<size_t>(size_in_bytes)))
        return true;
    }
  }

  if (heap()->ShouldExpandOldGenerationOnSlowAllocation() && Expand()) {
    DCHECK((CountTotalPages() > 1) ||
           (static_cast<size_t>(size_in_bytes) <= free_list_.Available()));
    return RefillLinearAllocationAreaFromFreeList(
        static_cast<size_t>(size_in_bytes));
  }

  // If sweeper threads are active, wait for them at that point and steal
  // elements form their free-lists. Allocation may still fail their which
  // would indicate that there is not enough memory for the given allocation.
  return SweepAndRetryAllocation(size_in_bytes);
}
