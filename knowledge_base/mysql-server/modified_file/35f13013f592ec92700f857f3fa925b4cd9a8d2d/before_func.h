   */
  template <class Func, class ProtoAllocator>
  void post(Func &&f, const ProtoAllocator &a) const {
    io_ctx_->defer_work(std::forward<Func>(f), a);
  }

  /**
   * defer function call for later execution.
   *
   * Effect:
   *
   * The executor:
   *
   * - SHALL NOT block forward progress of the caller pending completion of f().
   * - SHOULD NOT begin f()'s progress before the call to defer()
   *   completes.
   */
  template <class Func, class ProtoAllocator>
  void defer(Func &&f, const ProtoAllocator &a) const {
    post(std::forward<Func>(f), a);
  }

 private:
  friend io_context;

  explicit executor_type(io_context &ctx) : io_ctx_{std::addressof(ctx)} {}

  io_context *io_ctx_{nullptr};
};

inline bool operator==(const io_context::executor_type &a,
                       const io_context::executor_type &b) noexcept {
  return std::addressof(a.context()) == std::addressof(b.context());
}
inline bool operator!=(const io_context::executor_type &a,
                       const io_context::executor_type &b) noexcept {
  return !(a == b);
}

// io_context::executor_type is an executor even though it doesn't have an
// default constructor
template <>
struct is_executor<io_context::executor_type> : std::true_type {};

inline io_context::executor_type io_context::get_executor() noexcept {
  return executor_type(*this);
}

/**
 * cancel all async-ops of a file-descriptor.
 */
inline stdx::expected<void, std::error_code> io_context::cancel(
    native_handle_type fd) {
  bool need_notify{false};
  {
    // check all async-ops
    std::lock_guard<std::mutex> lk(mtx_);

    while (auto op = active_ops_.extract_first(fd)) {
      op->cancel();

      cancelled_ops_.push_back(std::move(op));

      need_notify = true;
    }
  }

  // wakeup the loop to deliver the cancelled fds
  if (true || need_notify) {
    io_service_->remove_fd(fd);

    notify_io_service_if_not_running_in_this_thread();
  }

  return {};
}

template <class Clock, class Duration>
inline io_context::count_type io_context::do_one_until(
    std::unique_lock<std::mutex> &lk,
    const std::chrono::time_point<Clock, Duration> &abs_time) {
  using namespace std::chrono_literals;

  const auto rel_time = abs_time - std::chrono::steady_clock::now();
  auto rel_time_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(rel_time);

  if (rel_time_ms < 0ms) {
    // expired already.
    rel_time_ms = 0ms;
  } else if (rel_time_ms < rel_time) {
    // std::chrono::ceil()
    rel_time_ms += 1ms;
  }

  return do_one(lk, rel_time_ms);
}

inline void io_context::notify_io_service_if_not_running_in_this_thread() {
  if (impl::Callstack<io_context>::contains(this) == nullptr) {
    io_service_->notify();
  }
}

// precond: lk MUST be locked
inline io_context::count_type io_context::do_one(
    std::unique_lock<std::mutex> &lk, std::chrono::milliseconds timeout) {
  impl::Callstack<io_context>::Context ctx(this);

  timer_queue_base *timer_q{nullptr};

  monitor mon(*this);

  if (!has_outstanding_work()) {
    wake_one_runner_(lk);
    return 0;
  }

  while (true) {
    // 1. deferred work.
    // 2. timer
    // 3. triggered events.

    // timer (2nd round)
    if (timer_q) {
      if (timer_q->run_one()) {
        wake_one_runner_(lk);
        return 1;
      } else {
        timer_q = nullptr;
      }
    }

    // deferred work
    if (deferred_work_.run_one()) {
      wake_one_runner_(lk);
      return 1;
    }

    // timer
    std::chrono::milliseconds min_duration{0};
    {
      std::lock_guard<std::mutex> lk(mtx_);
      // check the smallest timestamp of all timer-queues
      for (auto q : timer_queues_) {
        const auto duration = q->next();

        if (duration == duration.zero()) {
          timer_q = q;
          min_duration = duration;
          break;
        } else if ((duration != duration.max()) &&
