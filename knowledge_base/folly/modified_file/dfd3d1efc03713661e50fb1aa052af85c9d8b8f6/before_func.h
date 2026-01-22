  counters_.decrement(token.epoch_);
}

template <typename Tag>
template <typename T>
void rcu_domain<Tag>::call(T&& cbin) {
  auto node = new list_node;
  node->cb_ = [node, cb = std::forward<T>(cbin)]() {
    cb();
    delete node;
  };
  retire(node);
}

template <typename Tag>
void rcu_domain<Tag>::retire(list_node* node) noexcept {
  q_.push(node);

  // Note that it's likely we hold a read lock here,
  // so we can only half_sync(false).  half_sync(true)
  // or a synchronize() call might block forever.
  uint64_t time = std::chrono::duration_cast<std::chrono::milliseconds>(
