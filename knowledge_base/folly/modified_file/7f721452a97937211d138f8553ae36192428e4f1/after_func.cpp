    auto wlock = ulock.moveFromUpgradeToWrite();
    if (it->second && it->second->hasCallback()) {
      it->second->onUnset();
      wlock->callbackData_.erase(it->second.get());
    }

    requestData = std::move(it->second);
    wlock->requestData_.erase(it);
  }
}

namespace {
// Execute functor exec for all RequestData in data, which are not in other
// Similar to std::set_difference but avoid intermediate data structure
template <typename TData, typename TExec>
void exec_set_difference(const TData& data, const TData& other, TExec&& exec) {
  auto diter = data.begin();
  auto dend = data.end();
  auto oiter = other.begin();
  auto oend = other.end();
  while (diter != dend) {
    // Order of "if" optimizes for the 2 common cases:
    // 1) empty other, switching to default context
    // 2) identical other, switching to similar context with same callbacks
