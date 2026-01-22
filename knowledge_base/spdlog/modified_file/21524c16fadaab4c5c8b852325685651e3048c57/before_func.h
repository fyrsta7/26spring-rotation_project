class e_formatter SPDLOG_FINAL : public flag_formatter
{
    void format(const details::log_msg &msg, const std::tm &, fmt::memory_buffer &dest) override
    {
        auto duration = msg.time.time_since_epoch();
        auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() % 1000;
        fmt_helper::pad3(static_cast<int>(millis), dest);
    }
};
