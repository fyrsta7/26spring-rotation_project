
int TimeZone::dayOfMonth(Date_t date) const {
    auto time = getTimelibTime(date);
    return time->d;
}

int TimeZone::isoDayOfWeek(Date_t date) const {
    auto time = getTimelibTime(date);
    return timelib_iso_day_of_week(time->y, time->m, time->d);
}

