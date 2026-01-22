void RuntimeScheduler_Modern::scheduleWork(RawCallback&& callback) noexcept {
  SystraceSection s("RuntimeScheduler::scheduleWork");
  scheduleTask(SchedulerPriority::ImmediatePriority, std::move(callback));
}

std::shared_ptr<Task> RuntimeScheduler_Modern::scheduleTask(
    SchedulerPriority priority,
    jsi::Function&& callback) noexcept {
  SystraceSection s(
      "RuntimeScheduler::scheduleTask",
      "priority",
      serialize(priority),
      "callbackType",
      "jsi::Function");

  auto expirationTime = now_() + timeoutForSchedulerPriority(priority);
  auto task =
      std::make_shared<Task>(priority, std::move(callback), expirationTime);
