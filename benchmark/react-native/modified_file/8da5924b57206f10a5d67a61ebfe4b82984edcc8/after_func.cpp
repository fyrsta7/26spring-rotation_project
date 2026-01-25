void RuntimeScheduler_Modern::scheduleTask(std::shared_ptr<Task> task) {
  bool shouldScheduleEventLoop = false;

  {
    std::unique_lock lock(schedulingMutex_);

    // We only need to schedule the event loop if the task we're about to
    // schedule is the only one in the queue.
    // Otherwise, we don't need to schedule it because there's another one
    // running already that will pick up the new task.
    if (taskQueue_.empty() && !isEventLoopScheduled_) {
      isEventLoopScheduled_ = true;
      shouldScheduleEventLoop = true;
    }

    taskQueue_.push(std::move(task));
  }

  if (shouldScheduleEventLoop) {
    scheduleEventLoop();
  }
}