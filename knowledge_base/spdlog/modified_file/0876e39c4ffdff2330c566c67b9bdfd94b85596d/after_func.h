template<size_t Buffer_Size>
inline void pad3(int n, fmt::basic_memory_buffer<char, Buffer_Size> &dest)
{
    if (n > 999)
    {
        append_int(n, dest);
        return;
    }

    if (n > 99) // 100-999
    {
        dest.push_back(static_cast<char>('0' + n / 100));
        pad2(n % 100, dest);
        return;
    }
    if (n > 9) // 10-99
    {
        dest.push_back('0');
        dest.push_back(static_cast<char>('0' + n / 10));
        dest.push_back(static_cast<char>('0' + n % 10));
        return;
    }
    if (n >= 0)
    {
        dest.push_back('0');
        dest.push_back('0');
        dest.push_back(static_cast<char>('0' + n));
        return;
    }
    // negatives (unlikely, but just in case let fmt deal with it)
    fmt::format_to(dest, "{:03}", n);
}
