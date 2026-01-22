void CEventLoop::wait_for_event(WaitMode mode)
{
    fd_set rfds;
    fd_set wfds;
    FD_ZERO(&rfds);
    FD_ZERO(&wfds);

    int max_fd = 0;
    auto add_fd_to_set = [&max_fd](int fd, fd_set& set) {
        FD_SET(fd, &set);
        if (fd > max_fd)
            max_fd = fd;
    };

    int max_fd_added = -1;
    add_file_descriptors_for_select(rfds, max_fd_added);
    max_fd = max(max_fd, max_fd_added);
    for (auto& notifier : *s_notifiers) {
        if (notifier->event_mask() & CNotifier::Read)
            add_fd_to_set(notifier->fd(), rfds);
        if (notifier->event_mask() & CNotifier::Write)
            add_fd_to_set(notifier->fd(), wfds);
        if (notifier->event_mask() & CNotifier::Exceptional)
            ASSERT_NOT_REACHED();
    }

    bool queued_events_is_empty;
    {
        LOCKER(m_lock);
        queued_events_is_empty = m_queued_events.is_empty();
    }

    timeval now;
    struct timeval timeout = { 0, 0 };
    bool should_wait_forever = false;
    if (mode == WaitMode::WaitForEvents) {
        if (!s_timers->is_empty() && queued_events_is_empty) {
            gettimeofday(&now, nullptr);
            get_next_timer_expiration(timeout);
            AK::timeval_sub(&timeout, &now, &timeout);
        } else {
            should_wait_forever = true;
        }
    } else {
        should_wait_forever = false;
    }

    int marked_fd_count = select(max_fd + 1, &rfds, &wfds, nullptr, should_wait_forever ? nullptr : &timeout);
    if (marked_fd_count < 0) {
        ASSERT_NOT_REACHED();
    }

    if (!s_timers->is_empty()) {
        gettimeofday(&now, nullptr);
    }

    for (auto& it : *s_timers) {
        auto& timer = *it.value;
        if (!timer.has_expired(now))
            continue;
#ifdef CEVENTLOOP_DEBUG
        dbgprintf("CEventLoop: Timer %d has expired, sending CTimerEvent to %p\n", timer.timer_id, timer.owner);
#endif
        post_event(*timer.owner, make<CTimerEvent>(timer.timer_id));
        if (timer.should_reload) {
            timer.reload(now);
        } else {
            // FIXME: Support removing expired timers that don't want to reload.
            ASSERT_NOT_REACHED();
        }
    }

    if (!marked_fd_count)
        return;

    for (auto& notifier : *s_notifiers) {
        if (FD_ISSET(notifier->fd(), &rfds)) {
            if (notifier->on_ready_to_read)
                notifier->on_ready_to_read();
        }
        if (FD_ISSET(notifier->fd(), &wfds)) {
            if (notifier->on_ready_to_write)
                notifier->on_ready_to_write();
        }
    }

    process_file_descriptors_after_select(rfds);
}
