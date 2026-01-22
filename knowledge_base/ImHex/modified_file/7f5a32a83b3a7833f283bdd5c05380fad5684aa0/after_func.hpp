    template<typename ... Args>
    inline std::string format(const char *format, Args ... args) {
        ssize_t size = snprintf( nullptr, 0, format, args ... );

        if (size <= 0)
            return "";

        std::vector<char> buffer(size + 1, 0x00);
        snprintf(buffer.data(), size + 1, format, args ...);

        return std::string(buffer.data(), buffer.data() + size);
    }
