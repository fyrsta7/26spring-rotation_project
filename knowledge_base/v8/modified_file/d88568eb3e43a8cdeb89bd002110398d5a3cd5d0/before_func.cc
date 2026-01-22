void CompilationStateImpl::WaitForCompilationEvent(
    CompilationEvent expect_event) {
  auto compilation_event_semaphore = std::make_shared<base::Semaphore>(0);
  AddCallback(
      [compilation_event_semaphore, expect_event](CompilationEvent event) {
        if (event == expect_event ||
            event == CompilationEvent::kFailedCompilation) {
          compilation_event_semaphore->Signal();
        }
      });

  constexpr JobDelegate* kNoDelegate = nullptr;
  ExecuteCompilationUnits(background_compile_token_, async_counters_.get(),
                          kNoDelegate, kBaselineOnly);
  compilation_event_semaphore->Wait();
}
