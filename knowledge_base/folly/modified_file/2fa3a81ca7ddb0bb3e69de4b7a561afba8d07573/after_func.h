
#include <folly/Optional.h>
#include <folly/experimental/coro/Baton.h>
#include <folly/experimental/coro/Coroutine.h>
#include <folly/experimental/coro/Invoke.h>
#include <folly/experimental/coro/Task.h>
#include <folly/experimental/coro/Traits.h>
#include <folly/experimental/coro/detail/Helpers.h>
#include <folly/futures/Future.h>

#if FOLLY_HAS_COROUTINES

namespace folly {
namespace coro {
template <typename Awaitable>
Task<Optional<lift_unit_t<detail::decay_rvalue_reference_t<
    detail::lift_lvalue_reference_t<semi_await_result_t<Awaitable>>>>>>
timed_wait(Awaitable awaitable, Duration duration) {
  Baton baton;
  Try<lift_unit_t<detail::decay_rvalue_reference_t<
      detail::lift_lvalue_reference_t<semi_await_result_t<Awaitable>>>>>
      result;

  Executor* executor = co_await co_current_executor;
  auto sleepFuture = futures::sleep(duration).toUnsafeFuture();
  auto posted = new std::atomic<bool>(false);
  sleepFuture.setCallback_(
      [posted, &baton, executor = Executor::KeepAlive<>{executor}](
          auto&&, auto&&) {
        if (!posted->exchange(true, std::memory_order_acq_rel)) {
          executor->add([&baton] { baton.post(); });
        } else {
          delete posted;
        }
      });

  {
    auto t = co_invoke(
        [awaitable = std::move(
             awaitable)]() mutable -> Task<semi_await_result_t<Awaitable>> {
          co_return co_await std::move(awaitable);
        });
    std::move(t).scheduleOn(executor).start(
        [posted, &baton, &result, sleepFuture = std::move(sleepFuture)](
            auto&& r) mutable {
          if (!posted->exchange(true, std::memory_order_acq_rel)) {
            result = std::move(r);
            baton.post();
