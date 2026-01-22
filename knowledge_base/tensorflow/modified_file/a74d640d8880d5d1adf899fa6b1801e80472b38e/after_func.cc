      unsigned total_size = 0;
      for (int i = 0; i < non_blocking_work_sharding_factor_; ++i) {
        total_size += non_blocking_work_queues_[i]->queue.Size();
      }
      return total_size;
    }
  }

  int64 GetTracemeId() { return traceme_id_.load(std::memory_order_relaxed); }

  void SetTracemeId(int64 value) { traceme_id_ = value; }
  void SetRank(int64 value) { rank_ = value; }

  void SetWaiter(uint64 version, Waiter* waiter, mutex* mutex) {
    {
      tf_shared_lock lock(run_handler_waiter_mu_);
      // Most of the request won't change sub pool for recomputation.
      // Optimization for avoiding holding exclusive lock to reduce contention.
      if (sub_thread_pool_waiter_ == waiter) {
