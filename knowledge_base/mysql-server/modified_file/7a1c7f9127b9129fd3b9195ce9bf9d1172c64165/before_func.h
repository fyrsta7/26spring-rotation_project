    size_t ndx = fd_events_processed_;

    auto ev = fd_events_[ndx];

    // if there are multiple events:
    // - OUT before IN.
    // - IN before ERR|HUP.
    // - ERR before HUP.
    short revent{};
    if (ev.events & EPOLLOUT) {
      fd_events_[ndx].events &= ~EPOLLOUT;
      revent = EPOLLOUT;
    } else if (ev.events & EPOLLIN) {
      fd_events_[ndx].events &= ~EPOLLIN;
      revent = EPOLLIN;
    } else if (ev.events & EPOLLERR) {
      fd_events_[ndx].events &= ~EPOLLERR;
      revent = EPOLLERR;
    } else if (ev.events & EPOLLHUP) {
      fd_events_[ndx].events &= ~EPOLLHUP;
      revent = EPOLLHUP;
    }

    // all interesting events processed, go the next one.
    if ((fd_events_[ndx].events & (EPOLLIN | EPOLLOUT | EPOLLERR | EPOLLHUP)) ==
        0) {
      fd_events_processed_++;
    }

    return fd_event{ev.data.fd, revent};
  }

  stdx::expected<fd_event, std::error_code> update_fd_events(
