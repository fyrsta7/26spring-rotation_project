
BencString *BencString::Decode(const char *bytes, size_t *lenOut)
{
    if (!bytes || !ChrIsDigit(*bytes))
        return NULL;

    int64_t len;
    const char *start = ParseBencInt(bytes, len);
    if (!start || *start != ':' || len < 0)
        return NULL;

    start++;
    if (memchr(start, '\0', (size_t)len))
        return NULL;

    if (lenOut)
        *lenOut = (start - bytes) + (size_t)len;
    return new BencString(start, (size_t)len);
