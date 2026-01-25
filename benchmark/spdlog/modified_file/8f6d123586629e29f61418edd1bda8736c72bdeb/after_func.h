SPDLOG_INLINE spdlog::level::level_enum from_str(const std::string &name) SPDLOG_NOEXCEPT
{
    auto it = std::find(std::begin(level_string_views), std::end(level_string_views), name);
    if (it != std::end(level_string_views))
        return static_cast<level::level_enum>(it - std::begin(level_string_views));

    // check also for "warn" and "err" before giving up..
    if (name == "warn")
    {
        return level::warn;
    }
    if (name == "err")
    {
        return level::err;
    }
    return level::off;
}