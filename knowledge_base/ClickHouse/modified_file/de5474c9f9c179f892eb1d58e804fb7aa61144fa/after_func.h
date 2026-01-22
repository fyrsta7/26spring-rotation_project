
    using Searcher = std::conditional_t<case_insensitive, VolnitskyCaseInsensitiveUTF8, VolnitskyUTF8>;

    static void vectorConstant(
        const ColumnString::Chars & haystack_data,
        const ColumnString::Offsets & haystack_offsets,
        const String & needle,
        [[maybe_unused]] const ColumnPtr & start_pos_,
        PaddedPODArray<UInt8> & res)
    {
        const size_t haystack_size = haystack_offsets.size();

        assert(haystack_size == res.size());
        assert(start_pos_ == nullptr);

        if (haystack_offsets.empty())
            return;

        /// Fast path for [I]LIKE, because the result is always true or false
        /// col [i]like '%%'
        /// col not [i]like '%%'
        /// col like '%'
        /// col not [i]like '%'
        /// match(like, '^$')
        if ((is_like && (needle == "%%" or needle == "%")) || (!is_like && needle == ".*"))
        {
            for (auto & re : res)
                re = !negate;
            return;
        }

        /// Special case that the [I]LIKE expression reduces to finding a substring in a string
        String strstr_pattern;
        if (is_like && impl::likePatternIsSubstring(needle, strstr_pattern))
        {
            const UInt8 * const begin = haystack_data.data();
            const UInt8 * const end = haystack_data.data() + haystack_data.size();
            const UInt8 * pos = begin;

            /// The current index in the array of strings.
            size_t i = 0;

            /// TODO You need to make that `searcher` is common to all the calls of the function.
            Searcher searcher(strstr_pattern.data(), strstr_pattern.size(), end - pos);

            /// We will search for the next occurrence in all rows at once.
            while (pos < end && end != (pos = searcher.search(pos, end - pos)))
            {
                /// Let's determine which index it refers to.
                while (begin + haystack_offsets[i] <= pos)
                {
                    res[i] = negate;
                    ++i;
                }

                /// We check that the entry does not pass through the boundaries of strings.
                if (pos + strstr_pattern.size() < begin + haystack_offsets[i])
                    res[i] = !negate;
                else
                    res[i] = negate;

                pos = begin + haystack_offsets[i];
                ++i;
            }

            /// Tail, in which there can be no substring.
            if (i < res.size())
                memset(&res[i], negate, (res.size() - i) * sizeof(res[0]));

            return;
        }

        const auto & regexp = Regexps::Regexp(Regexps::createRegexp<is_like, /*no_capture*/ true, case_insensitive>(needle));

        String required_substring;
        bool is_trivial;
        bool required_substring_is_prefix; /// for `anchored` execution of the regexp.

        regexp.getAnalyzeResult(required_substring, is_trivial, required_substring_is_prefix);

        if (required_substring.empty())
        {
            if (!regexp.getRE2()) /// An empty regexp. Always matches.
                memset(res.data(), !negate, haystack_size * sizeof(res[0]));
            else
            {
                size_t prev_offset = 0;
                for (size_t i = 0; i < haystack_size; ++i)
                {
                    const bool match = regexp.getRE2()->Match(
                            {reinterpret_cast<const char *>(&haystack_data[prev_offset]), haystack_offsets[i] - prev_offset - 1},
                            0,
                            haystack_offsets[i] - prev_offset - 1,
                            re2_st::RE2::UNANCHORED,
                            nullptr,
                            0);
                    res[i] = negate ^ match;

                    prev_offset = haystack_offsets[i];
                }
            }
        }
        else
        {
            /// NOTE This almost matches with the case of impl::likePatternIsSubstring.

            const UInt8 * const begin = haystack_data.data();
            const UInt8 * const end = haystack_data.begin() + haystack_data.size();
            const UInt8 * pos = begin;

            /// The current index in the array of strings.
            size_t i = 0;

            Searcher searcher(required_substring.data(), required_substring.size(), end - pos);

            /// We will search for the next occurrence in all rows at once.
            while (pos < end && end != (pos = searcher.search(pos, end - pos)))
            {
                /// Determine which index it refers to.
                while (begin + haystack_offsets[i] <= pos)
                {
                    res[i] = negate;
                    ++i;
                }

                /// We check that the entry does not pass through the boundaries of strings.
                if (pos + required_substring.size() < begin + haystack_offsets[i])
                {
                    /// And if it does not, if necessary, we check the regexp.
                    if (is_trivial)
                        res[i] = !negate;
                    else
                    {
                        const char * str_data = reinterpret_cast<const char *>(&haystack_data[haystack_offsets[i - 1]]);
                        size_t str_size = haystack_offsets[i] - haystack_offsets[i - 1] - 1;

                        /** Even in the case of `required_substring_is_prefix` use UNANCHORED check for regexp,
                          *  so that it can match when `required_substring` occurs into the string several times,
                          *  and at the first occurrence, the regexp is not a match.
                          */
                        const size_t start_pos = (required_substring_is_prefix) ? (reinterpret_cast<const char *>(pos) - str_data) : 0;
                        const size_t end_pos = str_size;

                        const bool match = regexp.getRE2()->Match(
                                {str_data, str_size},
                                start_pos,
                                end_pos,
                                re2_st::RE2::UNANCHORED,
                                nullptr,
                                0);
                        res[i] = negate ^ match;
                    }
                }
                else
                    res[i] = negate;

                pos = begin + haystack_offsets[i];
                ++i;
            }
