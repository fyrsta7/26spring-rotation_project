      unsigned total_size = 0;
      for (int i = 0; i < non_blocking_work_sharding_factor_; ++i) {
        total_size += non_blocking_work_queues_[i]->queue.Size();
      }
      return total_size;
    }
  }

  int64 GetTracemeId() { return traceme_id_.load(std::memory_order_relaxed); }
