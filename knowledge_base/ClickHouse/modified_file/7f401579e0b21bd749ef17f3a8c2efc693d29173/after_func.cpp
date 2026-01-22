template <typename T>
void compressDataForType(const char * source, UInt32 source_size, char * dest)
{
    if (source_size % sizeof(T) != 0)
        throw Exception(ErrorCodes::CANNOT_COMPRESS, "Cannot GCD compress, data size {}  is not aligned to {}", source_size, sizeof(T));

    const char * const source_end = source + source_size;

    T gcd{};
    const auto * cur_source = source;
    while (cur_source < source_end)
    {
        if (cur_source == source)
        {
            gcd = unalignedLoad<T>(cur_source);
        }
        else
        {
            gcd = gcd_func<T>(gcd, unalignedLoad<T>(cur_source));
        }
        if (gcd == T(1)) {
            break;
        }
    }

    unalignedStore<T>(dest, gcd);
    dest += sizeof(T);

    cur_source = source;
    while (cur_source < source_end)
    {
        unalignedStore<T>(dest, unalignedLoad<T>(cur_source) / gcd);
        cur_source += sizeof(T);
        dest += sizeof(T);
    }
}
