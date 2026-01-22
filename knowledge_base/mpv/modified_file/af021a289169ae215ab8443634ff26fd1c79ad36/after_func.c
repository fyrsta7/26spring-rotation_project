        }

        uint64_t now_monotonic = ts.tv_sec * 1000000LL + ts.tv_nsec / 1000;
        uint64_t ust_mp_time = mp_time_us() - (now_monotonic - wl->sync[index].ust);
        wl->last_sbc_mp_time = ust_mp_time;
    }

    if (!wl->sync[index].sbc)
        return;

    wl->last_queue_display_time = wl->last_sbc_mp_time + sbc_passed*wl->vsync_duration;
}

void vo_wayland_wakeup(struct vo *vo)
{
    struct vo_wayland_state *wl = vo->wl;
    (void)write(wl->wakeup_pipe[1], &(char){0}, 1);
}

void vo_wayland_wait_frame(struct vo_wayland_state *wl)
{
    struct pollfd fds[1] = {
        {.fd = wl->display_fd,     .events = POLLIN },
    };

    double vblank_time = 1e6 / wl->current_output->refresh_rate;
    int64_t finish_time = mp_time_us() + vblank_time;

    while (wl->frame_wait && finish_time > mp_time_us()) {

        int poll_time = ceil((double)(finish_time - mp_time_us()) / 1000);
        if (poll_time < 0) {
            poll_time = 0;
        }

        while (wl_display_prepare_read(wl->display) != 0)
            wl_display_dispatch_pending(wl->display);
        wl_display_flush(wl->display);

        poll(fds, 1, poll_time);

        wl_display_read_events(wl->display);
        wl_display_roundtrip(wl->display);
    }

    if (wl->frame_wait) {
