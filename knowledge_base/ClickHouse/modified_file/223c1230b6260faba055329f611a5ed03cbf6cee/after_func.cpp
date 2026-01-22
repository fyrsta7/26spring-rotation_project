void ThreadStatus::detachQuery(bool exit_if_already_detached, bool thread_exits)
{
    LockMemoryExceptionInThread lock_memory_tracker(VariableContext::Global);

    if (exit_if_already_detached && thread_state == ThreadState::DetachedFromQuery)
    {
        thread_state = thread_exits ? ThreadState::Died : ThreadState::DetachedFromQuery;
        return;
    }

    assertState({ThreadState::AttachedToQuery}, __PRETTY_FUNCTION__);

    finalizeQueryProfiler();
    finalizePerformanceCounters();

    /// Detach from thread group
    {
        std::lock_guard guard(thread_group->mutex);
        thread_group->threads.erase(this);
    }
    performance_counters.setParent(&ProfileEvents::global_counters);
    memory_tracker.reset();

    memory_tracker.setParent(thread_group->memory_tracker.getParent());

    query_id.clear();
    query_context.reset();

    /// Avoid leaking of ThreadGroupStatus::finished_threads_counters_memory
    /// (this is in case someone uses system thread but did not call getProfileEventsCountersAndMemoryForThreads())
    {
        std::lock_guard guard(thread_group->mutex);
        auto stats = std::move(thread_group->finished_threads_counters_memory);
    }

    thread_group.reset();

    thread_state = thread_exits ? ThreadState::Died : ThreadState::DetachedFromQuery;

#if defined(OS_LINUX)
    if (os_thread_priority)
    {
        LOG_TRACE(log, "Resetting nice");

        if (0 != setpriority(PRIO_PROCESS, thread_id, 0))
            LOG_ERROR(log, "Cannot 'setpriority' back to zero: {}", errnoToString());

        os_thread_priority = 0;
    }
#endif
}
