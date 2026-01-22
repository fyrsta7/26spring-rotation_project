
pair<void*, std::size_t> IOBufQueue::preallocateSlow(
    std::size_t min, std::size_t newAllocationSize, std::size_t max) {
  // Avoid grabbing update guard, since we're manually setting the cache ptrs.
  flushCache();
  // Allocate a new buffer of the requested max size.
  unique_ptr<IOBuf> newBuf(IOBuf::create(std::max(min, newAllocationSize)));

  tailStart_ = newBuf->writableTail();
  cachePtr_->cachedRange = std::pair<uint8_t*, uint8_t*>(
      tailStart_, tailStart_ + newBuf->tailroom());
  reusableTail_ = newBuf.get();
  appendToChain(head_, std::move(newBuf), false);
  return make_pair(writableTail(), std::min<std::size_t>(max, tailroom()));
}

void IOBufQueue::maybeReuseTail() {
  if (reusableTail_ == nullptr || reusableTail_->isSharedOne() ||
      // Includes the reusableTail_ == head_->prev() case.
      reusableTail_->tailroom() <= head_->prev()->tailroom()) {
    return;
  }
  std::unique_ptr<IOBuf> newTail;
  if (reusableTail_->length() == 0) {
    // Nothing was written to the old tail, we can just move it to the end.
    if (reusableTail_ == head_.get()) {
      newTail = std::exchange(head_, head_->pop());
    } else {
      newTail = reusableTail_->unlink();
    }
  } else {
    auto freeFn = [](void*, void* p) { delete reinterpret_cast<IOBuf*>(p); };
    // We know the tail is not shared, so we can clone it and wrap it in a
    // new (unshared) IOBuf that owns its writable tail to reuse it.

    // For the case when we're already dealing with a reused tail, we can
    // use its parent IOBuf to avoid chaining IOBuf objects in the destructor.
