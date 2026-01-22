void ThreadStatusUtil::TEST_StateDelay(const ThreadStatus::StateType state) {
  auto delay = states_delay[state].load(std::memory_order_relaxed);
  if (delay > 0) {
    Env::Default()->SleepForMicroseconds(delay);
  }
}
