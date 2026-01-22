    static void vector(const ColumnString::Chars & data,
        const ColumnString::Offsets & offsets,
        ColumnString::Chars & res_data,
        ColumnString::Offsets & res_offsets)
    {
        UErrorCode err = U_ZERO_ERROR;

        const UNormalizer2 *normalizer = NormalizeImpl::getNormalizer(&err);
        if (U_FAILURE(err))
            throw Exception(ErrorCodes::CANNOT_NORMALIZE_STRING, "Normalization failed (getNormalizer): {}", u_errorName(err));

        size_t size = offsets.size();
        res_offsets.resize(size);

        ColumnString::Offset current_from_offset = 0;
        ColumnString::Offset current_to_offset = 0;

        PODArray<UChar> from_uchars;
        PODArray<UChar> to_uchars;

        for (size_t i = 0; i < size; ++i)
        {
            size_t from_size = offsets[i] - current_from_offset - 1;

            from_uchars.resize(from_size + 1);
            int32_t from_code_points = 0;
            u_strFromUTF8(
                from_uchars.data(),
                from_uchars.size(),
                &from_code_points,
                reinterpret_cast<const char*>(&data[current_from_offset]),
                from_size,
                &err);
            if (U_FAILURE(err))
                throw Exception(ErrorCodes::CANNOT_NORMALIZE_STRING, "Normalization failed (strFromUTF8): {}", u_errorName(err));

            to_uchars.resize(from_code_points * NormalizeImpl::expansionFactor + 1);

            int32_t to_code_points = unorm2_normalize(
                normalizer,
                from_uchars.data(),
                from_code_points,
                to_uchars.data(),
                to_uchars.size(),
                &err);
            if (U_FAILURE(err))
                throw Exception(ErrorCodes::CANNOT_NORMALIZE_STRING, "Normalization failed (normalize): {}", u_errorName(err));

            size_t max_to_size = current_to_offset + 4 * to_code_points + 1;
            if (res_data.size() < max_to_size)
                res_data.resize(max_to_size);

            int32_t to_size = 0;
            u_strToUTF8(
                reinterpret_cast<char*>(&res_data[current_to_offset]),
                res_data.size() - current_to_offset,
                &to_size,
                to_uchars.data(),
                to_code_points,
                &err);
            if (U_FAILURE(err))
                throw Exception(ErrorCodes::CANNOT_NORMALIZE_STRING, "Normalization failed (strToUTF8): {}", u_errorName(err));

            current_to_offset += to_size;
            res_data[current_to_offset] = 0;
            ++current_to_offset;
            res_offsets[i] = current_to_offset;

            current_from_offset = offsets[i];
        }

        res_data.resize(current_to_offset);
    }
