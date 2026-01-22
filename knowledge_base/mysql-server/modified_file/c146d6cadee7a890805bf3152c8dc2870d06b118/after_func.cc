static double seconds_used;
static steady_clock::time_point timer_start;
static size_t bytes_processed = 0;

void StartBenchmarkTiming() {
  assert(!timer_running);
  timer_running = true;
  timer_start = steady_clock::now();
}

void StopBenchmarkTiming() {
  if (timer_running) {
    auto used = steady_clock::now() - timer_start;
    seconds_used += duration<double>(used).count();
    timer_running = false;
  }
}

void SetBytesProcessed(size_t bytes) { bytes_processed = bytes; }

void internal_do_microbenchmark(const char *name, void (*func)(size_t)) {
#if !defined(NDEBUG)
  printf(
      "WARNING: Running microbenchmark in debug mode. "
      "Timings will be misleading.\n");

  // There's no point in timing in debug mode, so just run 10 times
  // so that we don't waste build time (this should give us enough runs
  // to verify that we don't crash).
  seconds_used = 0.0;
  size_t num_iterations = 10;
  StartBenchmarkTiming();
  func(num_iterations);
  StopBenchmarkTiming();
#else
  // Do 100 iterations as rough calibration. (Often, this will over- or
  // undershoot by as much as 50%, but that's fine.)
  static constexpr size_t calibration_iterations = 100;
  seconds_used = 0.0;
  StartBenchmarkTiming();
  func(calibration_iterations);
  StopBenchmarkTiming();
  double seconds_used_per_iteration = seconds_used / calibration_iterations;

  // Scale so that we end up around one second per benchmark
  // (but never less than 100).
  size_t num_iterations =
      std::max<size_t>(lrint(1.0 / seconds_used_per_iteration), 100);

  // Do the actual run.
  seconds_used = 0.0;
