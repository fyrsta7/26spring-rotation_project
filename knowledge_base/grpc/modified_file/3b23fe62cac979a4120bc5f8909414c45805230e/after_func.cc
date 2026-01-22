void WorkStealingThreadPool::WorkStealingThreadPoolImpl::Run(
    EventEngine::Closure* closure) {
  GPR_DEBUG_ASSERT(quiesced_.load(std::memory_order_relaxed) == false);
  if (g_local_queue != nullptr) {
    g_local_queue->Add(closure);
  } else {
    queue_.Add(closure);
  }
  // Signal a worker in any case, even if work was added to a local queue. This
  // improves performance on 32-core streaming benchmarks with small payloads.
  work_signal_.Signal();
}
