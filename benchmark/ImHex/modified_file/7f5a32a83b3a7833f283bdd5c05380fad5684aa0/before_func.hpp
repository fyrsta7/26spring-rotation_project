    template<typename ... Args>
    inline std::string format(const std::string &format, Args ... args) {
        ssize_t size = snprintf( nullptr, 0, format.c_str(), args ... );

        if( size <= 0 )
            return "";

        std::vector<char> buffer(size + 1, 0x00);
        snprintf(buffer.data(), size + 1, format.c_str(), args ...);

        return std::string(buffer.data(), buffer.data() + size);
    }