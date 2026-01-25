// https://pubs.opengroup.org/onlinepubs/9699919799/functions/memmove.html
void* memmove(void* dest, void const* src, size_t n)
{
    if (dest < src)
        return memcpy(dest, src, n);
    if (((FlatPtr)dest - (FlatPtr)src) >= n)
        return memcpy(dest, src, n);

    u8* pd = (u8*)dest;
    u8 const* ps = (u8 const*)src;
    for (pd += n, ps += n; n--;)
        *--pd = *--ps;
    return dest;
}