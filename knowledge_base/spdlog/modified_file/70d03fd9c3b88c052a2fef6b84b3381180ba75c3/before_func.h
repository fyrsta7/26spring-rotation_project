    void pad_it(size_t count)
    {
        // count = std::min(count, spaces_.size());
        assert(count <= spaces_.size());
        fmt_helper::append_string_view(string_view_t(spaces_.data(), count), dest_);
    }
