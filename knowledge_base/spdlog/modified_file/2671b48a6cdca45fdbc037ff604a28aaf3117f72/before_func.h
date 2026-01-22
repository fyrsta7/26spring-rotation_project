    void format(const details::log_msg &msg, const std::tm &, fmt::memory_buffer &dest) override
    {
        if (padinfo_.enabled())
        {
            scoped_pad p(*msg.logger_name, padinfo_, dest);
            fmt_helper::append_string_view(*msg.logger_name, dest);
        }
        else
        {
            fmt_helper::append_string_view(*msg.logger_name, dest);
        }
    }
