  void wake() {
    std::unique_lock<std::mutex> nodeLock(mutex_);
    signaled_ = true;
    cond_.notify_one();
  }