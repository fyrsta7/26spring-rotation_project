    count += vec.size();
  }
  if (count) {
    std::vector<double> values;
    values.reserve(count);
    for (const auto& vec : valuesVec) {
      values.insert(values.end(), vec.begin(), vec.end());
    }
    DigestT digest(digestSize_);
    digests.push_back(digest.merge(values));
  }
  return DigestT::merge(digests);
}

template <typename DigestT>
void DigestBuilder<DigestT>::append(double value) {
  const auto numBuffers = cpuLocalBuffers_.size();
  auto cpuLocalBuf =
      &cpuLocalBuffers_[AccessSpreader<>::cachedCurrent(numBuffers)];
  std::unique_lock<SpinLock> g(cpuLocalBuf->mutex, std::try_to_lock);
  if (FOLLY_UNLIKELY(!g.owns_lock())) {
    // If the mutex is already held by another thread, either build() is
    // running, or this or that thread have a stale stripe (possibly because the
    // thread migrated right after the call to cachedCurrent()). So invalidate
    // the cache and wait on the mutex.
    AccessSpreader<>::invalidateCachedCurrent();
