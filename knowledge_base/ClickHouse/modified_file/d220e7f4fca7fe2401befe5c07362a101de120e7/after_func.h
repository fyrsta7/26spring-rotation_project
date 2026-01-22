
public:
    template <typename CharT>
    requires (sizeof(CharT) == 1)
    StringSearcher(const CharT * needle_, const size_t needle_size_)
        : needle{reinterpret_cast<const uint8_t *>(needle_)}, needle_size{needle_size_}
    {
        if (0 == needle_size)
            return;

        UTF8SequenceBuffer l_seq;
        UTF8SequenceBuffer u_seq;

        if (*needle < 0x80u)
        {
            first_needle_symbol_is_ascii = true;
            l = std::tolower(*needle);
            u = std::toupper(*needle);
        }
        else
        {
            auto first_u32 = UTF8::convertUTF8ToCodePoint(needle, needle_size);

            /// Invalid UTF-8
            if (!first_u32)
            {
                /// Process it verbatim as a sequence of bytes.
                size_t src_len = UTF8::seqLength(*needle);

                memcpy(l_seq, needle, src_len);
                memcpy(u_seq, needle, src_len);
            }
            else
            {
                uint32_t first_l_u32 = Poco::Unicode::toLower(*first_u32);
                uint32_t first_u_u32 = Poco::Unicode::toUpper(*first_u32);

                /// lower and uppercase variants of the first octet of the first character in `needle`
                size_t length_l = UTF8::convertCodePointToUTF8(first_l_u32, l_seq, sizeof(l_seq));
                size_t length_u = UTF8::convertCodePointToUTF8(first_u_u32, u_seq, sizeof(u_seq));

                if (length_l != length_u)
                    force_fallback = true;
            }

            l = l_seq[0];
            u = u_seq[0];

            if (force_fallback)
                return;
        }

#ifdef __SSE4_1__
        /// for detecting leftmost position of the first symbol
        patl = _mm_set1_epi8(l);
        patu = _mm_set1_epi8(u);
        /// lower and uppercase vectors of first 16 octets of `needle`

        const auto * needle_pos = needle;

        for (size_t i = 0; i < n;)
        {
            if (needle_pos == needle_end)
            {
                cachel = _mm_srli_si128(cachel, 1);
                cacheu = _mm_srli_si128(cacheu, 1);
                ++i;

                continue;
            }

            size_t src_len = std::min<size_t>(needle_end - needle_pos, UTF8::seqLength(*needle_pos));
            auto c_u32 = UTF8::convertUTF8ToCodePoint(needle_pos, src_len);

            if (c_u32)
            {
                int c_l_u32 = Poco::Unicode::toLower(*c_u32);
                int c_u_u32 = Poco::Unicode::toUpper(*c_u32);

                size_t dst_l_len = UTF8::convertCodePointToUTF8(c_l_u32, l_seq, sizeof(l_seq));
                size_t dst_u_len = UTF8::convertCodePointToUTF8(c_u_u32, u_seq, sizeof(u_seq));

                /// @note Unicode standard states it is a rare but possible occasion
                if (!(dst_l_len == dst_u_len && dst_u_len == src_len))
                {
                    force_fallback = true;
                    return;
                }
            }

            cache_actual_len += src_len;
            if (cache_actual_len < n)
                cache_valid_len += src_len;

            for (size_t j = 0; j < src_len && i < n; ++j, ++i)
            {
                cachel = _mm_srli_si128(cachel, 1);
                cacheu = _mm_srli_si128(cacheu, 1);

                if (needle_pos != needle_end)
                {
                    cachel = _mm_insert_epi8(cachel, l_seq[j], n - 1);
                    cacheu = _mm_insert_epi8(cacheu, u_seq[j], n - 1);

                    cachemask |= 1 << i;
                    ++needle_pos;
                }
            }
        }
