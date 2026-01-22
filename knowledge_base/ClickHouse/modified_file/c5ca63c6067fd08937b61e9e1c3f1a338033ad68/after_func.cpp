

template <> void inline copy<16>(UInt8 * dst, const UInt8 * src) { copy16(dst, src); }
template <> void inline wildCopy<16>(UInt8 * dst, const UInt8 * src, UInt8 * dst_end) { wildCopy16(dst, src, dst_end); }
template <> void inline copyOverlap<16, false>(UInt8 * op, const UInt8 *& match, const size_t offset) { copyOverlap16(op, match, offset); }
template <> void inline copyOverlap<16, true>(UInt8 * op, const UInt8 *& match, const size_t offset) { copyOverlap16Shuffle(op, match, offset); }


inline void copy32(UInt8 * dst, const UInt8 * src)
{
    /// There was an AVX here but with mash with SSE instructions, we got a big slowdown.
#if defined(__SSE2__)
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst),
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(src)));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + 16),
        _mm_loadu_si128(reinterpret_cast<const __m128i *>(src + 16)));
#else
    memcpy(dst, src, 16);
    memcpy(dst + 16, src + 16, 16);
#endif
}

inline void wildCopy32(UInt8 * dst, const UInt8 * src, const UInt8 * dst_end)
{
    /// Unrolling with clang is doing >10% performance degrade.
#if defined(__clang__)
    #pragma nounroll
#endif
    do
    {
        copy32(dst, src);
        dst += 32;
        src += 32;
    } while (dst < dst_end);
}

inline void copyOverlap32(UInt8 * op, const UInt8 *& match, const size_t offset)
{
    /// 4 % n.
    static constexpr int shift1[]
        = { 0,  1,  2,  1,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4 };

    /// 8 % n - 4 % n
    static constexpr int shift2[]
        = { 0,  0,  0,  1,  0, -1, -2, -3, -4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4 };

    /// 16 % n - 8 % n
    static constexpr int shift3[]
        = { 0,  0,  0, -1,  0, -2,  2,  1,  8, -1, -2, -3, -4, -5, -6, -7,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8 };

    /// 32 % n - 16 % n
    static constexpr int shift4[]
        = { 0,  0,  0,  1,  0,  1, -2,  2,  0, -2, -4,  5,  4,  3,  2,  1,  0, -1, -2, -3, -4, -5, -6, -7, -8, -9,-10,-11,-12,-13,-14,-15 };

    op[0] = match[0];
    op[1] = match[1];
    op[2] = match[2];
    op[3] = match[3];

    match += shift1[offset];
    memcpy(op + 4, match, 4);
    match += shift2[offset];
    memcpy(op + 8, match, 8);
    match += shift3[offset];
    memcpy(op + 16, match, 16);
    match += shift4[offset];
}


template <> void inline copy<32>(UInt8 * dst, const UInt8 * src) { copy32(dst, src); }
template <> void inline wildCopy<32>(UInt8 * dst, const UInt8 * src, UInt8 * dst_end) { wildCopy32(dst, src, dst_end); }
template <> void inline copyOverlap<32, false>(UInt8 * op, const UInt8 *& match, const size_t offset) { copyOverlap32(op, match, offset); }


/// See also https://stackoverflow.com/a/30669632

template <size_t copy_amount, bool use_shuffle>
bool NO_INLINE decompressImpl(
     const char * const source,
     char * const dest,
     size_t source_size,
     size_t dest_size)
{
    const UInt8 * ip = reinterpret_cast<const UInt8 *>(source);
    UInt8 * op = reinterpret_cast<UInt8 *>(dest);
    const UInt8 * const input_end = ip + source_size;
    UInt8 * const output_begin = op;
    UInt8 * const output_end = op + dest_size;

    /// Unrolling with clang is doing >10% performance degrade.
#if defined(__clang__)
    #pragma nounroll
#endif
    while (true)
    {
        size_t length;

        auto continue_read_length = [&]
        {
            unsigned s;
            do
            {
                s = *ip++;
                length += s;
            } while (unlikely(s == 255 && ip < input_end));
        };

        /// Get literal length.

        if (unlikely(ip >= input_end))
            return false;

        const unsigned token = *ip++;
        length = token >> 4;
        if (length == 0x0F)
        {
            if (unlikely(ip + 1 >= input_end))
                return false;
            continue_read_length();
        }

        /// Copy literals.

        UInt8 * copy_end = op + length;

        /// input: Hello, world
        ///        ^-ip
        /// output: xyz
        ///            ^-op  ^-copy_end
        /// output: xyzHello, w
        ///                   ^- excessive copied bytes due to "wildCopy"
        /// input: Hello, world
        ///              ^-ip
        /// output: xyzHello, w
        ///                  ^-op (we will overwrite excessive bytes on next iteration)

        if (unlikely(copy_end > output_end))
            return false;

        // Due to implementation specifics the copy length is always a multiple of copy_amount
        size_t real_length = 0;
        if constexpr (copy_amount == 8)
            real_length = (((length >> 3) + 1) * 8);
        else if constexpr (copy_amount == 16)
            real_length = (((length >> 4) + 1) * 16);
        else if constexpr (copy_amount == 32)
            real_length = (((length >> 5) + 1) * 32);
        else
