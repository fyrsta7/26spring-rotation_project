        WCHAR *string;
        uint32_t hash;

        Item(WCHAR *string=NULL, uint32_t hash=0) : string(string), hash(hash) { }
    };

    Vec<Item> items;
    size_t count;
    Allocator *allocator;

    // variation of CRC-32 which deals with strings that are
    // mostly ASCII and should be treated case independently
    static uint32_t GetQuickHashI(const WCHAR *str) {
        uint32_t crc = 0;
        for (WCHAR c; (c = *str); str++) {
            if ((c & 0xFF80))
                c = '\x80';
            else if ('A' <= c && c <= 'Z')
