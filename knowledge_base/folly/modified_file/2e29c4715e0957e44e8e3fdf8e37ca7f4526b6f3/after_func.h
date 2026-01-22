  template <typename Clock, typename Duration>
  std::cv_status wait(std::chrono::time_point<Clock, Duration> deadline) {
    std::cv_status status = std::cv_status::no_timeout;
    std::unique_lock<std::mutex> nodeLock(mutex_);
    while (!signaled_ && status != std::cv_status::timeout) {
