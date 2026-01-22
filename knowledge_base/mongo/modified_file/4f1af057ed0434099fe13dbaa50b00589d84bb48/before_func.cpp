    failWithErrno(errno);
}

/**
 * Takes a Date_t deadline and sets the appropriate values in a timespec structure.
 */
void tsFromDate(const Date_t& deadline, struct timespec& ts) {
    ts.tv_sec = deadline.toTimeT();
    ts.tv_nsec = (deadline.toMillisSinceEpoch() % 1000) * 1'000'000;
}
}  // namespace

TicketHolder::TicketHolder(int num) : _outof(num) {
    check(sem_init(&_sem, 0, num));
}

TicketHolder::~TicketHolder() {
    check(sem_destroy(&_sem));
}

bool TicketHolder::tryAcquire() {
    while (0 != sem_trywait(&_sem)) {
        if (errno == EAGAIN)
            return false;
        if (errno != EINTR)
            failWithErrno(errno);
    }
    return true;
}

