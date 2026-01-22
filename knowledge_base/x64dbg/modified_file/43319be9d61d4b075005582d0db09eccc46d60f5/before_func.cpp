String StringUtils::ToCompressedHex(unsigned char* buffer, size_t size)
{
    if(!size)
        return "";
    String result;
    result.reserve(size * 2);
    for(size_t i = 0; i < size;)
    {
        size_t repeat = 0;
        auto lastCh = buffer[i];
        result.push_back(HEXLOOKUP[(lastCh >> 4) & 0xF]);
        result.push_back(HEXLOOKUP[lastCh & 0xF]);
        for(; i < size && buffer[i] == lastCh; i++)
            repeat++;
        result.append(StringUtils::sprintf("{%" fext "X}", repeat));
    }
    return result;
}
