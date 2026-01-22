void MarkCompactCollector::SweepSpaces() {
  GCTracer::Scope gc_scope(tracer_, GCTracer::Scope::MC_SWEEP);
#ifdef DEBUG
  state_ = SWEEP_SPACES;
#endif
  SweeperType how_to_sweep =
      FLAG_lazy_sweeping ? LAZY_CONSERVATIVE : CONSERVATIVE;
  if (AreSweeperThreadsActivated()) how_to_sweep = PARALLEL_CONSERVATIVE;
  if (FLAG_expose_gc) how_to_sweep = CONSERVATIVE;
  if (sweep_precisely_) how_to_sweep = PRECISE;
  // Noncompacting collections simply sweep the spaces to clear the mark
  // bits and free the nonlive blocks (for old and map spaces).  We sweep
  // the map space last because freeing non-live maps overwrites them and
  // the other spaces rely on possibly non-live maps to get the sizes for
  // non-live objects.

  SweepSpace(heap()->old_pointer_space(), how_to_sweep);
  SweepSpace(heap()->old_data_space(), how_to_sweep);

  RemoveDeadInvalidatedCode();
  SweepSpace(heap()->code_space(), PRECISE);

  SweepSpace(heap()->cell_space(), PRECISE);

  EvacuateNewSpaceAndCandidates();

  if (how_to_sweep == PARALLEL_CONSERVATIVE) {
    // TODO(hpayer): The starting of the sweeper threads should be after
    // SweepSpace old data space.
    StartSweeperThreads();
    if (FLAG_parallel_sweeping && !FLAG_concurrent_sweeping) {
      WaitUntilSweepingCompleted();
    }
  }

  // ClearNonLiveTransitions depends on precise sweeping of map space to
  // detect whether unmarked map became dead in this collection or in one
  // of the previous ones.
  SweepSpace(heap()->map_space(), PRECISE);

  // Deallocate unmarked objects and clear marked bits for marked objects.
  heap_->lo_space()->FreeUnmarkedObjects();
}
