    void format(const details::log_msg &, const std::tm &tm_time, fmt::memory_buffer &dest) override
    {
        // fmt::format_to(dest, "{} {} {} ", days[tm_time.tm_wday],
        // months[tm_time.tm_mon], tm_time.tm_mday);
        // date
        fmt_helper::append_str(days[tm_time.tm_wday], dest);
        dest.push_back(' ');
        fmt_helper::append_str(months[tm_time.tm_mon], dest);
        dest.push_back(' ');
        fmt_helper::append_int(tm_time.tm_mday, dest);
        dest.push_back(' ');
        // time

        fmt_helper::pad2(tm_time.tm_hour, dest);
        dest.push_back(':');
        fmt_helper::pad2(tm_time.tm_min, dest);
        dest.push_back(':');
        fmt_helper::pad2(tm_time.tm_sec, dest);
        dest.push_back(' ');
        fmt_helper::append_int(tm_time.tm_year + 1900, dest);
    }
