std::string Capitalize(std::string str)
{
    if (str.empty()) return str;
    str[0] = ToUpper(str.front());
    return str;
}

std::string HexStr(const Span<const uint8_t> s)
{
    std::string rv;
    static constexpr char hexmap[16] = { '0', '1', '2', '3', '4', '5', '6', '7',
                                         '8', '9', 'a', 'b', 'c', 'd', 'e', 'f' };
