    return ret;
}

static void esp_vfs_log_fd_set(const char *fds_name, const fd_set *fds)
{
    if (fds_name && fds) {
        ESP_LOGD(TAG, "FDs in %s =", fds_name);
        for (int i = 0; i < MAX_FDS; ++i) {
            if (esp_vfs_safe_fd_isset(i, fds)) {
                ESP_LOGD(TAG, "%d", i);
            }
        }
    }
}

int esp_vfs_select(int nfds, fd_set *readfds, fd_set *writefds, fd_set *errorfds, struct timeval *timeout)
{
    // NOTE: Please see the "Synchronous input/output multiplexing" section of the ESP-IDF Programming Guide
    // (API Reference -> Storage -> Virtual Filesystem) for a general overview of the implementation of VFS select().
    int ret = 0;
    struct _reent* r = __getreent();

    ESP_LOGD(TAG, "esp_vfs_select starts with nfds = %d", nfds);
    if (timeout) {
        ESP_LOGD(TAG, "timeout is %lds + %ldus", (long)timeout->tv_sec, timeout->tv_usec);
    }
    esp_vfs_log_fd_set("readfds", readfds);
    esp_vfs_log_fd_set("writefds", writefds);
    esp_vfs_log_fd_set("errorfds", errorfds);

    if (nfds > MAX_FDS || nfds < 0) {
        ESP_LOGD(TAG, "incorrect nfds");
        __errno_r(r) = EINVAL;
        return -1;
    }

    // Capture s_vfs_count to a local variable in case a new driver is registered or removed during this actual select()
    // call. s_vfs_count cannot be protected with a mutex during a select() call (which can be one without a timeout)
    // because that could block the registration of new driver.
    const size_t vfs_count = s_vfs_count;
    fds_triple_t *vfs_fds_triple;
    if ((vfs_fds_triple = calloc(vfs_count, sizeof(fds_triple_t))) == NULL) {
        __errno_r(r) = ENOMEM;
        ESP_LOGD(TAG, "calloc is unsuccessful");
        return -1;
    }

    esp_vfs_select_sem_t sel_sem = {
        .is_sem_local = false,
        .sem = NULL,
    };

    int (*socket_select)(int, fd_set *, fd_set *, fd_set *, struct timeval *) = NULL;
    for (int fd = 0; fd < nfds; ++fd) {
        _lock_acquire(&s_fd_table_lock);
        const bool is_socket_fd = s_fd_table[fd].permanent;
        const int vfs_index = s_fd_table[fd].vfs_index;
        const int local_fd = s_fd_table[fd].local_fd;
        if (esp_vfs_safe_fd_isset(fd, errorfds)) {
            s_fd_table[fd].has_pending_select = true;
        }
        _lock_release(&s_fd_table_lock);

        if (vfs_index < 0) {
            continue;
        }

        if (is_socket_fd) {
            if (!socket_select) {
                // no socket_select found yet so take a look
                if (esp_vfs_safe_fd_isset(fd, readfds) ||
                        esp_vfs_safe_fd_isset(fd, writefds) ||
                        esp_vfs_safe_fd_isset(fd, errorfds)) {
                    const vfs_entry_t *vfs = s_vfs[vfs_index];
                    socket_select = vfs->vfs.socket_select;
                    sel_sem.sem = vfs->vfs.get_socket_select_semaphore();
                }
            }
            continue;
        }

        fds_triple_t *item = &vfs_fds_triple[vfs_index]; // FD sets for VFS which belongs to fd
        if (esp_vfs_safe_fd_isset(fd, readfds)) {
            item->isset = true;
            FD_SET(local_fd, &item->readfds);
            FD_CLR(fd, readfds);
            ESP_LOGD(TAG, "removing %d from readfds and adding as local FD %d to fd_set of VFS ID %d", fd, local_fd, vfs_index);
        }
        if (esp_vfs_safe_fd_isset(fd, writefds)) {
            item->isset = true;
            FD_SET(local_fd, &item->writefds);
            FD_CLR(fd, writefds);
            ESP_LOGD(TAG, "removing %d from writefds and adding as local FD %d to fd_set of VFS ID %d", fd, local_fd, vfs_index);
        }
        if (esp_vfs_safe_fd_isset(fd, errorfds)) {
            item->isset = true;
            FD_SET(local_fd, &item->errorfds);
            FD_CLR(fd, errorfds);
            ESP_LOGD(TAG, "removing %d from errorfds and adding as local FD %d to fd_set of VFS ID %d", fd, local_fd, vfs_index);
        }
    }

    // all non-socket VFSs have their FD sets in vfs_fds_triple
    // the global readfds, writefds and errorfds contain only socket FDs (if
    // there any)

    if (!socket_select) {
        // There is no socket VFS registered or select() wasn't called for
        // any socket. Therefore, we will use our own signalization.
        sel_sem.is_sem_local = true;
        if ((sel_sem.sem = xSemaphoreCreateBinary()) == NULL) {
            free(vfs_fds_triple);
            __errno_r(r) = ENOMEM;
            ESP_LOGD(TAG, "cannot create select semaphore");
            return -1;
        }
    }

    void **driver_args = calloc(vfs_count, sizeof(void *));

    if (driver_args == NULL) {
        free(vfs_fds_triple);
        __errno_r(r) = ENOMEM;
        ESP_LOGD(TAG, "calloc is unsuccessful for driver args");
        return -1;
    }

    for (size_t i = 0; i < vfs_count; ++i) {
        const vfs_entry_t *vfs = get_vfs_for_index(i);
        fds_triple_t *item = &vfs_fds_triple[i];

        if (vfs && vfs->vfs.start_select && item->isset) {
            // call start_select for all non-socket VFSs with has at least one FD set in readfds, writefds, or errorfds
            // note: it can point to socket VFS but item->isset will be false for that
            ESP_LOGD(TAG, "calling start_select for VFS ID %d with the following local FDs", i);
            esp_vfs_log_fd_set("readfds", &item->readfds);
            esp_vfs_log_fd_set("writefds", &item->writefds);
            esp_vfs_log_fd_set("errorfds", &item->errorfds);
            esp_err_t err = vfs->vfs.start_select(nfds, &item->readfds, &item->writefds, &item->errorfds, sel_sem,
                    driver_args + i);

            if (err != ESP_OK) {
                call_end_selects(i, vfs_fds_triple, driver_args);
                (void) set_global_fd_sets(vfs_fds_triple, vfs_count, readfds, writefds, errorfds);
                if (sel_sem.is_sem_local && sel_sem.sem) {
                    vSemaphoreDelete(sel_sem.sem);
                    sel_sem.sem = NULL;
                }
                free(vfs_fds_triple);
                free(driver_args);
                __errno_r(r) = EINTR;
                ESP_LOGD(TAG, "start_select failed: %s", esp_err_to_name(err));
                return -1;
            }
        }
    }

    if (socket_select) {
        ESP_LOGD(TAG, "calling socket_select with the following FDs");
        esp_vfs_log_fd_set("readfds", readfds);
        esp_vfs_log_fd_set("writefds", writefds);
        esp_vfs_log_fd_set("errorfds", errorfds);
        ret = socket_select(nfds, readfds, writefds, errorfds, timeout);
        ESP_LOGD(TAG, "socket_select returned %d and the FDs are the following", ret);
        esp_vfs_log_fd_set("readfds", readfds);
        esp_vfs_log_fd_set("writefds", writefds);
        esp_vfs_log_fd_set("errorfds", errorfds);
    } else {
        if (readfds) {
            FD_ZERO(readfds);
        }
        if (writefds) {
            FD_ZERO(writefds);
        }
        if (errorfds) {
            FD_ZERO(errorfds);
        }

        TickType_t ticks_to_wait = portMAX_DELAY;
        if (timeout) {
            uint32_t timeout_ms = (timeout->tv_sec * 1000) + (timeout->tv_usec / 1000);
            /* Round up the number of ticks.
             * Not only we need to round up the number of ticks, but we also need to add 1.
             * Indeed, `select` function shall wait for AT LEAST timeout, but on FreeRTOS,
             * if we specify a timeout of 1 tick to `xSemaphoreTake`, it will take AT MOST
             * 1 tick before triggering a timeout. Thus, we need to pass 2 ticks as a timeout
             * to `xSemaphoreTake`. */
            ticks_to_wait = ((timeout_ms + portTICK_PERIOD_MS - 1) / portTICK_PERIOD_MS) + 1;
            ESP_LOGD(TAG, "timeout is %dms", timeout_ms);
        }
        ESP_LOGD(TAG, "waiting without calling socket_select");
        xSemaphoreTake(sel_sem.sem, ticks_to_wait);
    }

    call_end_selects(vfs_count, vfs_fds_triple, driver_args); // for VFSs for start_select was called before

    if (ret >= 0) {
        ret += set_global_fd_sets(vfs_fds_triple, vfs_count, readfds, writefds, errorfds);
    }
