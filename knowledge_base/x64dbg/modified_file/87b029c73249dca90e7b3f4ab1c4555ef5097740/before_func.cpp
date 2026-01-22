String StringUtils::sprintf(_Printf_format_string_ const char* format, ...)
{
    va_list args;
    va_start(args, format);
    std::vector<char> buffer(256, '\0');
    while(true)
    {
        int res = _vsnprintf_s(buffer.data(), buffer.size(), _TRUNCATE, format, args);
        if(res == -1)
        {
            buffer.resize(buffer.size() * 2);
            continue;
        }
        else
            break;
    }
    va_end(args);
    return String(buffer.data());
}
