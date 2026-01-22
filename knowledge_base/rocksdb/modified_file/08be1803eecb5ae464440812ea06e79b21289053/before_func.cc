void ThreadStatusUtil::TEST_StateDelay(
    const ThreadStatus::StateType state) {
  Env::Default()->SleepForMicroseconds(
      states_delay[state].load(std::memory_order_relaxed));
}
