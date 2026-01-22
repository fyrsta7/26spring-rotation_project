int ScavengerCollector::NumberOfScavengeTasks() {
  if (!FLAG_parallel_scavenge) return 1;
  const int num_scavenge_tasks =
      static_cast<int>(heap_->new_space()->TotalCapacity()) / MB + 1;
  static int num_cores = V8::GetCurrentPlatform()->NumberOfWorkerThreads() + 1;
  int tasks =
      Max(1, Min(Min(num_scavenge_tasks, kMaxScavengerTasks), num_cores));
  if (!heap_->CanExpandOldGeneration(
          static_cast<size_t>(tasks * Page::kPageSize))) {
    // Optimize for memory usage near the heap limit.
    tasks = 1;
  }
  return tasks;
}
