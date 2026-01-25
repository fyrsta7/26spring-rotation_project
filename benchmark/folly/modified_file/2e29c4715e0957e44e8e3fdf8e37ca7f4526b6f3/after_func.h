  void wake() {
    std::lock_guard<std::mutex> nodeLock(mutex_);
    signaled_ = true;
    cond_.notify_one();
  }