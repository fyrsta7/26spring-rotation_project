
  AsyncPipe& operator=(AsyncPipe&& pipe) {
    if (this != &pipe) {
      CHECK(!onClosed_ || onClosed_->wasInvokeRequested())
          << "If an onClosed callback is specified and the generator still "
          << "exists, the publisher must explicitly close the pipe prior to "
          << "destruction.";
      std::move(*this).close();
      queue_ = std::move(pipe.queue_);
      onClosed_ = std::move(pipe.onClosed_);
    }
    return *this;
  }

  static std::pair<folly::coro::AsyncGenerator<T&&>, AsyncPipe> create(
      folly::Function<void()> onClosed = nullptr) {
    auto queue = std::make_shared<Queue>();
    auto cancellationSource = std::optional<folly::CancellationSource>();
    auto onClosedCallback = std::unique_ptr<OnClosedCallback>();
    if (onClosed != nullptr) {
      cancellationSource.emplace();
      onClosedCallback = std::make_unique<OnClosedCallback>(
          *cancellationSource, std::move(onClosed));
    }
    auto guard =
        folly::makeGuard([cancellationSource = std::move(cancellationSource)] {
          if (cancellationSource) {
            cancellationSource->requestCancellation();
          }
        });
    return {
