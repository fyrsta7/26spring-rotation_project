 private:
  const SettingMetadata& meta_;
  const Type defaultValue_;

  SharedMutex globalLock_;
  std::shared_ptr<Contents> globalValue_;

