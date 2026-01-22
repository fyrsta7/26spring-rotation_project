void CompilationStateImpl::WaitForCompilationEvent(
    CompilationEvent expect_event) {
  auto compilation_event_semaphore = std::make_shared<base::Semaphore>(0);
  base::EnumSet<CompilationEvent> events{expect_event,
                                         CompilationEvent::kFailedCompilation};
  {
    base::MutexGuard callbacks_guard(&callbacks_mutex_);
    if (finished_events_.contains_any(events)) return;
    callbacks_.emplace_back(
        [compilation_event_semaphore, events](CompilationEvent event) {
          if (events.contains(event)) compilation_event_semaphore->Signal();
        });
  }

  constexpr JobDelegate* kNoDelegate = nullptr;
  ExecuteCompilationUnits(background_compile_token_, async_counters_.get(),
                          kNoDelegate, kBaselineOnly);
  compilation_event_semaphore->Wait();
}
