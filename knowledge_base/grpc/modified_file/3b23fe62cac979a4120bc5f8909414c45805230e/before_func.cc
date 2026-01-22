void WorkStealingThreadPool::WorkStealingThreadPoolImpl::Run(
    EventEngine::Closure* closure) {
  GPR_DEBUG_ASSERT(quiesced_.load(std::memory_order_relaxed) == false);
  if (g_local_queue != nullptr) {
    g_local_queue->Add(closure);
    return;
  }
  queue_.Add(closure);
  work_signal_.Signal();
}
