 private:
  const SettingMetadata& meta_;
  const Type defaultValue_;

  SharedMutex globalLock_;
  std::shared_ptr<Contents> globalValue_;

  /* Local versions start at 0, this will force a read on first local access. */
  CachelinePadded<std::atomic<size_t>> globalVersion_{1};

  ThreadLocal<CachelinePadded<
